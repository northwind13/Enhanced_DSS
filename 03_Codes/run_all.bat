@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   DisasterAware Simulation - Run All
echo ============================================
echo.

REM ---- 1) Python'u bul (python ya da py) ----
set "PY="
python --version >nul 2>&1 && set "PY=python"
if not defined PY (
    py --version >nul 2>&1 && set "PY=py"
)
if not defined PY (
    echo [HATA] Python bulunamadi.
    echo        python.org'dan kurun ve kurulumda "Add Python to PATH" secin.
    echo.
    pause
    exit /b 1
)
echo [OK] Python komutu: %PY%
echo.

REM ---- 2) Gerekli kutuphaneler: yoksa kur, varsa atla ----
for %%P in (numpy matplotlib) do (
    %PY% -c "import %%P" >nul 2>&1
    if errorlevel 1 (
        echo [KUR] %%P bulunamadi, yukleniyor...
        %PY% -m pip install %%P
        if errorlevel 1 (
            echo [HATA] %%P yuklenemedi. Internet baglantisini kontrol edin.
            pause
            exit /b 1
        )
    ) else (
        echo [OK] %%P zaten kurulu.
    )
)
echo.

REM ---- 3) Deneyleri sirayla calistir ----
echo --- exp_main.py (karsilastirma, olceklenme, kural-azaltma) ---
%PY% exp_main.py || (echo [HATA] exp_main.py & pause & exit /b 1)
echo.
echo --- exp_sens.py (duyarlilik taramalari) ---
%PY% exp_sens.py || (echo [HATA] exp_sens.py & pause & exit /b 1)
echo.
echo --- exp_maps.py (ortam katmanlari + uzaysal anlik goruntuler) ---
%PY% exp_maps.py || (echo [HATA] exp_maps.py & pause & exit /b 1)
echo.

REM ---- 4) Figurleri figures\ altina topla ----
if not exist "figures" mkdir "figures"
move /Y fig_sim_*.png "figures\" >nul 2>&1

echo ============================================
echo   BITTI.
echo   Figurler : figures\fig_sim_*.png
echo   Sonuclar : results.json
echo ============================================
echo.
pause
