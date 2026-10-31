@echo off
chcp 65001 > nul
cd /d "%~dp0"
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
python -u -m src.p2r_train_v2 --dataset nfunsw --tag p2r --plan stage1 --seeds 10 --threads 7
for %%S in (minmax signedlog_standard signedlog_minmax) do python -u -m src.p2r_train_v2 --dataset nfunsw --tag p2r_%%S --only m4:K4 --seeds 5 --threads 7
rem R1-11 one-hot protocol and bit-expanded flags
python -u -m src.p2r_train_v2 --dataset nfunsw --tag p2r_enc --only m4:K4 --seeds 5 --threads 7
echo ===== NF training done =====
