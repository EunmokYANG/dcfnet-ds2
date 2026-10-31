@echo off
cd /d "%~dp0"
rem two windows, 7 threads each; rerun skips finished runs (outputs\p2r\runs\*.json)
rem v2: the second window starts 20 s later so that the two conda activations do not race
start "p2r CIC" cmd /k "run_2a_train_cic_v1.bat >> logs\train_cic.txt 2>&1"
timeout /t 20 /nobreak > nul
start "p2r NF"  cmd /k "run_2b_train_nf_v1.bat  >> logs\train_nf.txt 2>&1"
echo started. progress: logs\train_cic.txt, logs\train_nf.txt
pause
