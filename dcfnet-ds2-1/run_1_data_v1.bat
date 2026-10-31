@echo off
chcp 65001 > nul
cd /d "%~dp0"
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
echo ===== 1. partitions (about 10-20 min) =====
python -u -m src.p2r_data_v2 --dataset ciciot2023 --tag p2r                                  > logs\data_cic_p2r.txt 2>&1
python -u -m src.p2r_data_v2 --dataset nfunsw     --tag p2r                                  > logs\data_nf_p2r.txt 2>&1
python -u -m src.p2r_data_v2 --dataset nfunsw     --tag p2r_enc    --categorical encode      > logs\data_nf_enc.txt 2>&1
for %%S in (minmax signedlog_standard signedlog_minmax) do (
  python -u -m src.p2r_data_v2 --dataset ciciot2023 --tag p2r_%%S --scaler %%S > logs\data_cic_%%S.txt 2>&1
  python -u -m src.p2r_data_v2 --dataset nfunsw     --tag p2r_%%S --scaler %%S > logs\data_nf_%%S.txt 2>&1
)
python -u -m src.p2r_data_v2 --dataset ciciot2023 --tag p2r_split2 --split-seed 2           > logs\data_cic_split2.txt 2>&1
python -u -m src.p2r_data_v2 --dataset ciciot2023 --tag p2r_split3 --split-seed 3           > logs\data_cic_split3.txt 2>&1
type logs\data_cic_p2r.txt
type logs\data_nf_p2r.txt
echo ===== 1 done =====
pause
