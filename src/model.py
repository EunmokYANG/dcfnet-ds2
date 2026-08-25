"""DCFNet-DS 모델 정의.

핵심 아이디어
  기존:  x_{l+1} = x0 * (W_l x_l) + x_l          <- 잔차로 차수가 누적 혼합
  제안:  u_k     = x0 * (W_k u_{k-1})            <- 잔차/편향 제거, 정확히 k차
         z       = sum_k alpha_k u_k,  sum alpha_k = 1

alpha_k 가 "탐지에 몇 차 상호작용이 필요한가"를 직접 알려주는 지표가 된다.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from src import config as C


# ------------------------------------------------------------------ loss
def focal_loss(gamma=C.FOCAL_GAMMA, alpha=C.FOCAL_ALPHA):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        return alpha * tf.pow(1.0 - pt, gamma) * ce

    return loss


# ------------------------------------------------------------------ layers
class GlobalDegreeLogits(layers.Layer):
    """입력과 무관한 전역 차수 로짓. 차수 순수성을 완전히 보존한다."""

    def __init__(self, max_degree, **kw):
        super().__init__(**kw)
        self.max_degree = max_degree

    def build(self, input_shape):
        self.logits = self.add_weight(
            name="global_logits", shape=(self.max_degree,),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        b = tf.shape(x)[0]
        return tf.tile(tf.expand_dims(self.logits, 0), [b, 1])

    def get_config(self):
        return {**super().get_config(), "max_degree": self.max_degree}


class UniformDegreeWeights(layers.Layer):
    """모든 차수에 1/K 를 부여. 게이트 없는 ablation 용."""

    def __init__(self, max_degree, **kw):
        super().__init__(**kw)
        self.max_degree = max_degree

    def call(self, x):
        b = tf.shape(x)[0]
        w = tf.ones((1, self.max_degree), dtype=x.dtype) / float(self.max_degree)
        return tf.tile(w, [b, 1])

    def get_config(self):
        return {**super().get_config(), "max_degree": self.max_degree}


class DegreeScale(layers.Layer):
    """u_k 를 RMS 로 나눈다.

    학습 중에는 배치 RMS 로 나누고 이동평균을 갱신한다.
    추론 시에는 이동평균(상수)으로 나누므로 동차성이 보존된다.
        u_k(cx) / s_k = c^k * (u_k(x) / s_k)
    평균 이동이 없으므로 BatchNormalization 과 달리 차수가 깨지지 않는다.
    """

    def __init__(self, momentum=0.99, eps=1e-6, **kw):
        super().__init__(**kw)
        self.momentum = momentum
        self.eps = eps

    def build(self, input_shape):
        self.s = self.add_weight(name="rms", shape=(),
                                 initializer="zeros", trainable=False)
        self.step = self.add_weight(name="step", shape=(),
                                    initializer="zeros", trainable=False)
        super().build(input_shape)

    def call(self, x, training=None):
        batch_rms = tf.sqrt(tf.reduce_mean(tf.square(x)) + self.eps)
        if training:
            self.s.assign(self.momentum * self.s
                          + (1.0 - self.momentum) * batch_rms)
            self.step.assign_add(1.0)
            scale = batch_rms
        else:
            # Adam 식 편향 보정. 초기 s=0 에서 출발해도 1스텝 뒤부터
            # 올바른 스케일이 되어 학습/추론 불일치가 생기지 않는다.
            debias = 1.0 - tf.pow(self.momentum, tf.maximum(self.step, 1.0))
            scale = self.s / tf.maximum(debias, self.eps)
        return x / (scale + self.eps)

    def get_config(self):
        return {**super().get_config(),
                "momentum": self.momentum, "eps": self.eps}


class BiasAdd(layers.Layer):
    """로짓 편향. 차수 기여와 분리해 두어 분해 항등식을 명시한다."""

    def build(self, input_shape):
        self.b = self.add_weight(name="bias", shape=(int(input_shape[-1]),),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x + self.b


class DegreeFusion(layers.Layer):
    """[alpha, u_1..u_K] -> sum_k alpha_k u_k,  또는 [a, u] -> a * u"""

    def call(self, inputs):
        if len(inputs) == 2 and int(inputs[0].shape[-1]) == 1:
            return inputs[0] * inputs[1]
        alpha, degs = inputs[0], inputs[1:]
        out = alpha[:, 0:1] * degs[0]
        for i in range(1, len(degs)):
            out = out + alpha[:, i:i + 1] * degs[i]
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[1]


# ------------------------------------------------------------------ blocks
def cross_layer(x0, x):
    """선행 연구와 동일한 잔차 교차 계층 (DCN-V2)."""
    xw = layers.Dense(x0.shape[-1], use_bias=False)(x)
    return layers.Add()([layers.Multiply()([x0, xw]), x])


def cross_layer_pure(x0, x, l2=1e-5, name=None):
    """잔차/편향 없는 교차 계층. 출력은 정확히 (입력차수 + 1)차."""
    xw = layers.Dense(
        x0.shape[-1], use_bias=False,
        kernel_regularizer=regularizers.l2(l2),
        name=None if name is None else f"{name}_w",
    )(x)
    return layers.Multiply(name=name)([x0, xw])


def degree_gate(x0, max_degree, mode="instance"):
    if mode == "uniform":
        return UniformDegreeWeights(max_degree, name="degree_alpha")(x0)
    if mode == "global":
        g = GlobalDegreeLogits(max_degree, name="degree_logits")(x0)
    else:
        g = layers.Dense(max_degree, name="degree_logits")(x0)
    return layers.Softmax(name="degree_alpha")(g)


# ------------------------------------------------------------------ model
def build_model(input_dim, num_classes, degree_separated=False,
                max_degree=C.MAX_DEGREE, gate_mode="instance"):
    inputs = layers.Input(shape=(input_dim,), name="inputs")

    # 공통 투영
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Branch 1 : MLP
    mlp = layers.Dense(512, activation="relu")(x)
    mlp = layers.BatchNormalization()(mlp)
    mlp = layers.Dropout(0.4)(mlp)
    mlp = layers.Dense(256, activation="relu")(mlp)
    mlp = layers.BatchNormalization()(mlp)
    mlp = layers.Dropout(0.3)(mlp)
    mlp = layers.Dense(128, activation="relu")(mlp)

    # Branch 2 : Cross
    if not degree_separated:
        cross = inputs
        for _ in range(3):
            cross = cross_layer(inputs, cross)
    else:
        degs = [inputs]                       # u_1 = x0
        u = inputs
        for k in range(2, max_degree + 1):
            u = cross_layer_pure(inputs, u, name=f"cross_deg{k}")
            degs.append(u)
        alpha = degree_gate(inputs, max_degree, gate_mode)
        cross = DegreeFusion(name="cross_degree_fusion")([alpha] + degs)

    # Fusion
    z = layers.Concatenate()([mlp, cross])
    z = layers.Dense(512, activation="relu")(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(256, activation="relu")(z)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(z)

    model = models.Model(inputs, outputs, name="dcfnet_ds")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=C.LEARNING_RATE),
        loss=focal_loss(),
        metrics=["accuracy"],
    )
    return model


def build_model_ds(input_dim, num_classes, max_degree=C.MAX_DEGREE,
                   l2=1e-5, use_gate=False, use_mlp=False):
    """DCFNet-DS : 차수 분리 + 선형 readout (제안 모델 M4).

        u_1 = x,   u_k = x * (W_k u~_{k-1}),   u~_k = u_k / s_k
        logit_c = sum_k (V_k u~_k)_c + b_c

    각 차수의 로짓 기여 phi_k = V_k u~_k 가 그대로 계산되므로
    sum_k phi_k + b = logit 이 항등식으로 성립한다.
    게이트가 없으므로 재매개변수화에 의한 식별 불가 문제도 없다.
    """
    inputs = layers.Input(shape=(input_dim,), name="inputs")

    parts, u_norm = [], None
    for k in range(1, max_degree + 1):
        u = inputs if k == 1 else cross_layer_pure(
            inputs, u_norm, l2=l2, name=f"cross_deg{k}")
        u_norm = DegreeScale(name=f"scale_deg{k}")(u)
        parts.append(layers.Dense(
            num_classes, use_bias=False,
            kernel_regularizer=regularizers.l2(l2),
            name=f"logit_deg{k}")(u_norm))

    if use_gate:                       # M4g : 식별 불가 실증용 ablation
        alpha = degree_gate(inputs, max_degree, "instance")
        parts = [DegreeFusion(name=f"gated_deg{k + 1}")(
            [alpha[:, k:k + 1], p]) for k, p in enumerate(parts)]

    if use_mlp:                        # M4m : 분해 붕괴 실증용 ablation
        m = layers.Dense(256, activation="relu")(inputs)
        m = layers.BatchNormalization()(m)
        m = layers.Dropout(0.3)(m)
        m = layers.Dense(128, activation="relu")(m)
        parts.append(layers.Dense(num_classes, use_bias=False,
                                  name="logit_mlp")(m))

    total = layers.Add(name="logit_sum")(parts) if len(parts) > 1 else parts[0]
    logits = BiasAdd(name="logits")(total)
    outputs = layers.Activation("softmax", name="probs")(logits)

    model = models.Model(inputs, outputs, name="dcfnet_ds")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=C.LEARNING_RATE),
        loss=focal_loss(), metrics=["accuracy"],
    )
    return model


def build_mlp(input_dim, num_classes):
    """신경망 baseline. 차수 경로 없이 MLP 만 사용한다."""
    inputs = layers.Input(shape=(input_dim,), name="inputs")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = models.Model(inputs, outputs, name="mlp_baseline")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=C.LEARNING_RATE),
        loss=focal_loss(), metrics=["accuracy"])
    return model


def build_variant(input_dim, num_classes, variant, max_degree=None):
    """config.VARIANTS 설정대로 모델을 만든다.

    max_degree 를 주면 config 값 대신 사용한다 (K 스윕용).
    """
    cfg = dict(C.VARIANTS[variant])
    if max_degree is not None:
        cfg["max_degree"] = max_degree
    if cfg.get("kind") == "mlp":
        return build_mlp(input_dim, num_classes)
    if cfg.get("kind") == "ds":
        return build_model_ds(
            input_dim, num_classes,
            max_degree=cfg.get("max_degree") or C.MAX_DEGREE,
            use_gate=cfg.get("use_gate", False),
            use_mlp=cfg.get("use_mlp", False))
    return build_model(
        input_dim, num_classes,
        degree_separated=cfg["degree_separated"],
        max_degree=cfg.get("max_degree") or C.MAX_DEGREE,
        gate_mode=cfg["gate_mode"] or "instance")


CUSTOM_OBJECTS = {
    "DegreeScale": DegreeScale,
    "BiasAdd": BiasAdd,
    "GlobalDegreeLogits": GlobalDegreeLogits,
    "UniformDegreeWeights": UniformDegreeWeights,
    "DegreeFusion": DegreeFusion,
    "loss": focal_loss(),
}
