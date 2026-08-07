@echo off
rem Start Friture.
rem
rem Two things this handles that a bare "python main.py" does not:
rem
rem 1. Finding an interpreter that has PyQt5. The python on the user PATH is
rem    the plain 3.11 install, which does not, so double-clicking this file
rem    would otherwise fail before it printed anything useful.
rem
rem 2. Telling Qt where its plugins are. This PyQt5 does not register them
rem    itself, which shows up as 'Could not find the Qt platform plugin
rem    "windows" in ""', then 'module "QtQuick.Controls" is not installed',
rem    then toolbar icons failing with 'Unsupported image format'. All three
rem    are Qt looking in an empty plugin path.
rem
rem Set FRITURE_PYTHON to override the interpreter choice.
rem
rem Keep this file ASCII-only: cmd reads .bat in the OEM code page, and the
rem paths above this directory are not ASCII.

setlocal

set "PYTHON="
if defined FRITURE_PYTHON call :try_python "%FRITURE_PYTHON%"
call :try_python "%~dp0.venv\Scripts\python.exe"
call :try_python "%~dp0..\ultraScan\.venv\Scripts\python.exe"
call :try_python "python"
call :try_python "py"

if not defined PYTHON goto :no_python

rem Via a file rather than for/f: nesting quotes inside for/f is its own
rem escaping puzzle, and the interpreter path may contain spaces.
set "QT_PROBE=%TEMP%\friture_qt_root.txt"
"%PYTHON%" -c "import PyQt5,os;print(os.path.dirname(PyQt5.__file__))" > "%QT_PROBE%" 2>nul
set "PYQT_DIR="
if exist "%QT_PROBE%" set /p PYQT_DIR=<"%QT_PROBE%"
del "%QT_PROBE%" >nul 2>&1

if not defined PYQT_DIR goto :no_qt
set "QT_ROOT=%PYQT_DIR%\Qt5"

rem the plugins root, not just platforms: imageformats is where the SVG
rem plugin lives, and the toolbar icons are SVGs
set "QT_PLUGIN_PATH=%QT_ROOT%\plugins"
set "QML2_IMPORT_PATH=%QT_ROOT%\qml"

"%PYTHON%" "%~dp0main.py" %*
if errorlevel 1 (
    echo.
    echo Friture exited with an error. Its log is at:
    rem asked of Python rather than spelled out: a Store-based interpreter has
    rem its %%LOCALAPPDATA%% redirected into the package's LocalCache, so the
    rem obvious path is not where the file actually lands
    "%PYTHON%" -c "import os,platformdirs;print(' ',os.path.join(platformdirs.user_log_dir('Friture',''),'friture.log.txt'))" 2>nul
    pause
)
exit /b 0

:try_python
rem First candidate that can import PyQt5 wins; the rest are not consulted.
if defined PYTHON exit /b 0
"%~1" -c "import PyQt5" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON=%~1"
exit /b 0

:no_python
echo Could not find a Python with PyQt5 installed.
echo.
echo Tried, in order:
if defined FRITURE_PYTHON echo   %FRITURE_PYTHON%   (from FRITURE_PYTHON)
echo   %~dp0.venv\Scripts\python.exe
echo   %~dp0..\ultraScan\.venv\Scripts\python.exe
echo   python           (whatever is on PATH)
echo   py
echo.
echo Either point FRITURE_PYTHON at the right interpreter, or make a venv
echo here:  python -m venv .venv ^&^& .venv\Scripts\pip install -e .
pause
exit /b 1

:no_qt
echo Found %PYTHON%, but could not work out where its Qt files are.
pause
exit /b 1
