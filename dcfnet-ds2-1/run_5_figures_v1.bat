@echo off
chcp 65001 > nul
cd /d "%~dp0"
call "C:\anaconda3\Scripts\activate.bat" dcfnet-ds
python -u -m src.p2r_figures_v1 --analysis outputs\p2r\analysis --out outputs\p2r\figures
echo ===== 5 done: outputs\p2r\figures =====
pause
