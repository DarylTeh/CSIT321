@echo off
setlocal

:: Initialize pythonchooser variable to "python"
set "pythonchooser=python"

:: Check if Python is available by running 'python --version'
echo Checking Python version with 'python --version'...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Changing python to py.
    set "pythonchooser=py"
) else (
    echo 'python' is available.
)

:: Set the installation directory
set "INSTALL_DIR=%USERPROFILE%\Downloads\HGR"

:: Clone the repository from GitHub
echo Cloning repository from GitHub...
git clone https://github.com/DarylTeh/CSIT321 "%INSTALL_DIR%" >nul 2>&1
if %errorlevel% neq 0 (
    echo Failed to clone repository. Exiting...
    exit /b %errorlevel%
)

:: Change to the cloned directory
cd /d "%INSTALL_DIR%"

:: Run the application setup
echo Installing dependencies...
%pythonchooser% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Exiting...
    exit /b %errorlevel%
)

:: Run the application
echo Running application...
%pythonchooser% app.py
