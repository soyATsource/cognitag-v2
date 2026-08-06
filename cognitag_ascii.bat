@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

:MENU
cls
echo ==========================================
echo   CogniTag v2  Dictionary Build Pipeline
echo ==========================================
echo.
echo   [ Setup ]
echo     0. Install required libraries
echo     9. Environment check (Python / Sudachi / Ollama)
echo.
echo   [ Validation ]
echo     C. Calibration test (prompt validity / approx 5 min)
echo     R. Rescue quarantined words (drop 1st trial, re-judge)
echo.
echo   [ Trial run ]
echo     1. Build corpus     (5 categories only)
echo     2. Extract candidates
echo     3. Facet annotation (30 words only)
echo.
echo   [ Full run ]
echo     4. Build corpus     (all 295 categories / approx 2 hours)
echo     5. Facet annotation (all candidates / long)
echo        NOTE: after 4, always run 2 before 5
echo     6. Run all          (4 -^> 2 -^> 5 in sequence)
echo.
echo   [ Inspect ]
echo     7. Status report
echo     8. Open output folder
echo.
echo     Q. Quit
echo.
set "SEL="
set /p SEL="Enter a number and press Enter: "

if /i "%SEL%"=="C" goto CALIBRATE
if /i "%SEL%"=="R" goto RESCUE
if /i "%SEL%"=="0" goto SETUP
if /i "%SEL%"=="9" goto CHECK
if /i "%SEL%"=="1" goto CORPUS_TEST
if /i "%SEL%"=="2" goto EXTRACT
if /i "%SEL%"=="3" goto ANNOTATE_TEST
if /i "%SEL%"=="4" goto CORPUS_FULL
if /i "%SEL%"=="5" goto ANNOTATE_FULL
if /i "%SEL%"=="6" goto RUN_ALL
if /i "%SEL%"=="7" goto STATS
if /i "%SEL%"=="8" goto OPENDIR
if /i "%SEL%"=="Q" goto END
goto MENU

:SETUP
cls
echo === Installing required libraries ===
echo.
python -m pip install sudachipy sudachidict_core ollama
echo.
echo Install finished.
pause
goto MENU

:CHECK
cls
echo === Environment check ===
echo.
echo [Python]
python --version
if errorlevel 1 (
  echo   Python not found. Check your PATH.
  pause
  goto MENU
)
echo.
echo [SudachiPy]
python -c "from sudachipy import Dictionary; d=Dictionary().create(); print('  OK ->', d.tokenize(chr(30690)+chr(30462))[0].dictionary_form())" 2>nul
if errorlevel 1 echo   Not installed. Run menu item 0.
echo.
echo [Ollama model list]
ollama list
if errorlevel 1 echo   ollama command not found.
echo.
echo [GPU usage] check this while generation is running
ollama ps
echo.
pause
goto MENU

:CALIBRATE
cls
echo === Calibration test ===
echo.
echo Uses the anchor words in facets.py (15 per axis) as ground truth
echo and measures whether the 4 axes are properly separated.
echo Always run this before a full annotation run.
echo.
set "MDL=gemma3:4b"
set /p MDL="Model [default gemma3:4b]: "
echo.
python run_pipeline.py calibrate --model %MDL%
echo.
pause
goto MENU

:RESCUE
cls
echo === Rescue quarantined words (drop 1st trial, re-judge) ===
echo.
echo Recomputes variance from the trial data already stored in
echo quarantine.json, excluding the first trial, and moves words
echo below the threshold into the dictionary.
echo No LLM calls are made.
echo.
set "THR=0.5"
set /p THR="Variance threshold [default 0.5]: "
echo.
echo --- Dry run first (no files are written) ---
echo.
python run_pipeline.py rescue --variance-threshold %THR% --dry-run
echo.
set "YN=N"
set /p YN="Apply for real? (Y/N) [default N]: "
if /i not "%YN%"=="Y" goto MENU
echo.
python run_pipeline.py rescue --variance-threshold %THR%
echo.
pause
goto MENU

:CORPUS_TEST
cls
echo === Build corpus (5 categories, trial) ===
echo.
set "MDL=gemma3:4b"
set /p MDL="Model [default gemma3:4b]: "
echo.
python run_pipeline.py corpus --limit 5 --model %MDL%
echo.
echo Inspect the generated files under the corpus folder.
echo Check that dialogue reads naturally and terms appear in context.
pause
goto MENU

:EXTRACT
cls
echo === Extract candidates and apply frequency filter ===
echo.
set "MINF=2"
set /p MINF="Minimum frequency [default 2]: "
echo.
python run_pipeline.py extract --min-frequency %MINF%
echo.
echo If the hapax ratio is too high, raise the minimum
echo frequency to 3 and run again.
pause
goto MENU

:ANNOTATE_TEST
cls
echo === Facet annotation (top 30 frequent words, trial) ===
echo.
set "MDL=gemma3:4b"
set /p MDL="Model [default gemma3:4b]: "
echo.
python run_pipeline.py annotate --limit 30 --model %MDL%
echo.
echo Note down the variance summary printed at the end.
echo It is the basis for choosing the threshold.
pause
goto MENU

:CORPUS_FULL
cls
echo === Build corpus (all 295 categories) ===
echo.
echo WARNING: this can take several hours.
echo          You can stop and resume with the same command.
echo.
set "YN=N"
set /p YN="Run it? (Y/N) [default N]: "
if /i not "%YN%"=="Y" goto MENU
set "MDL=gemma3:4b"
set /p MDL="Model [default gemma3:4b]: "
echo.
echo Start: %date% %time%
python run_pipeline.py corpus --model %MDL%
echo End:   %date% %time%
pause
goto MENU

:ANNOTATE_FULL
cls
echo === Facet annotation (all candidates) ===
echo.
echo Check: did you run candidate extraction (2) after building
echo        the corpus (4)? If not, you will annotate a stale list.
echo.
set "THR=0.5"
set /p THR="Variance threshold [default 0.5]: "
set "MDL=gemma3:4b"
set /p MDL="Model [default gemma3:4b]: "
set "ITV=0"
set /p ITV="Sleep seconds [0 recommended on PC / default 0]: "
echo.
echo Start: %date% %time%
python run_pipeline.py annotate --variance-threshold %THR% --model %MDL% --interval %ITV%
echo End:   %date% %time%
pause
goto MENU

:RUN_ALL
cls
echo === Run all (corpus -^> extract -^> annotate) ===
echo.
echo WARNING: the whole pipeline can take several hours to half a day.
echo          Running it overnight is recommended.
echo.
set "YN=N"
set /p YN="Run it? (Y/N) [default N]: "
if /i not "%YN%"=="Y" goto MENU
set "MDL=gemma3:4b"
set /p MDL="Model [default gemma3:4b]: "
set "THR=0.5"
set /p THR="Variance threshold [default 0.5]: "
echo.
echo ############ Start: %date% %time% ############
echo.
echo --- [1/3] Build corpus ---
python run_pipeline.py corpus --model %MDL%
echo.
echo --- [2/3] Extract candidates ---
python run_pipeline.py extract
echo.
echo --- [3/3] Facet annotation ---
python run_pipeline.py annotate --variance-threshold %THR% --model %MDL% --interval 0
echo.
echo ############ Done: %date% %time% ############
python run_pipeline.py stats
pause
goto MENU

:STATS
cls
python run_pipeline.py stats
echo.
pause
goto MENU

:OPENDIR
start "" "%~dp0"
goto MENU

:END
endlocal
exit /b 0
