@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem Waits until step 3 has written its last result (NF EBM seed 3), then runs 3b and 4 in sequence.
rem Start it in a NEW window while run_3_baselines_v2.bat is still running; it uses no CPU while waiting.
:wait
if exist outputs\p2r\baselines\nfunsw_p2r_ebm_s3.json goto go
echo %date% %time% waiting for step 3 (nfunsw_p2r_ebm_s3.json) ...
timeout /t 300 /nobreak > nul
goto wait
:go
echo step 3 finished; starting 3b in 2 minutes
timeout /t 120 /nobreak > nul
call run_3b_baselines_tuned_v1.bat < nul
call run_4_analysis_v2.bat < nul
echo ===== 3b and 4 done =====
pause
