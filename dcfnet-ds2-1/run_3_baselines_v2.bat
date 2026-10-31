@echo off
chcp 65001 > nul
cd /d "%~dp0"
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
rem v2: no pip install here. interpret is already installed; numpy must stay 1.26.4 for TensorFlow 2.16.2
python -c "import numpy, sys; v=numpy.__version__; print(\"numpy\", v); sys.exit(0 if v.startswith(\"1.26\") else 1)"
if errorlevel 1 ( echo numpy is not 1.26.x - run: python -m pip install "numpy==1.26.4" & pause & exit /b 1 )
python -u -m src.p2r_baselines_v1 --dataset ciciot2023 --tag p2r --seeds 3 > logs\baselines_cic.txt 2>&1
python -u -m src.p2r_baselines_v1 --dataset nfunsw     --tag p2r --seeds 3 > logs\baselines_nf.txt 2>&1
echo ===== 3 done =====
pause
