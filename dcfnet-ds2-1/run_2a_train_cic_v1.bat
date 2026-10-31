@echo off
chcp 65001 > nul
cd /d "%~dp0"
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
rem main protocol: 13 configurations x 10 seeds
python -u -m src.p2r_train_v2 --dataset ciciot2023 --tag p2r --plan stage1 --seeds 10 --threads 7
rem R2-2 alternative input maps, m4 K=4, 5 seeds
for %%S in (minmax signedlog_standard signedlog_minmax) do python -u -m src.p2r_train_v2 --dataset ciciot2023 --tag p2r_%%S --only m4:K4 --seeds 5 --threads 7
rem R1-13 repeated partitions, 5 seeds
for %%T in (p2r_split2 p2r_split3) do python -u -m src.p2r_train_v2 --dataset ciciot2023 --tag %%T --only m4:K4 nn_mlp --seeds 5 --threads 7
echo ===== CIC training done =====
