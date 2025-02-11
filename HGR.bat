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

:: Create the folder if it doesn’t exist
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: Move all extracted files to the target location
xcopy /E /Y "%~dp0*" "%INSTALL_DIR%"

:: Change to the target directory
cd /d "%INSTALL_DIR%"

::grant all permissions for folder
icacls "%INSTALL_DIR%" /grant "%USERPROFILE%":(OI)(CI)F /T

:: Run the application
echo Installing dependencies...
%pythonchooser% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Exiting...
    exit /b %errorlevel%
)

echo Running application...
%pythonchooser% app.py 