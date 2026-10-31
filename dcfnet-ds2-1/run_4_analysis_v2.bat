@echo off
chcp 65001 > nul
cd /d "%~dp0"
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
for %%D in (ciciot2023 nfunsw) do (
  python -u -m src.p2r_analysis_v2 all    --dataset %%D --tag p2r              > logs\analysis_%%D_p2r.txt 2>&1
  python -u -m src.p2r_analysis_v2 coeff  --dataset %%D --tag p2r --coeff-K 6 >> logs\analysis_%%D_p2r.txt 2>&1
  for %%S in (minmax signedlog_standard signedlog_minmax) do (
    python -u -m src.p2r_analysis_v2 shares --dataset %%D --tag p2r_%%S  > logs\analysis_%%D_%%S.txt 2>&1
    python -u -m src.p2r_analysis_v2 stats  --dataset %%D --tag p2r_%%S >> logs\analysis_%%D_%%S.txt 2>&1
  )
)
python -u -m src.p2r_analysis_v2 shares --dataset nfunsw --tag p2r_enc > logs\analysis_nf_enc.txt 2>&1
python -u -m src.p2r_analysis_v2 stats  --dataset nfunsw --tag p2r_enc >> logs\analysis_nf_enc.txt 2>&1
for %%T in (p2r_split2 p2r_split3) do python -u -m src.p2r_analysis_v2 stats --dataset ciciot2023 --tag %%T > logs\analysis_cic_%%T.txt 2>&1
echo ===== 4 done: outputs\p2r\analysis =====
pause
