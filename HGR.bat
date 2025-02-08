@echo off
setlocal

:: Set the installation directory
set "INSTALL_DIR=%USERPROFILE%\Downloads\HGR"

:: Create the folder if it doesn’t exist
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: Move all extracted files to the target location
xcopy /E /Y "%~dp0*" "%INSTALL_DIR%"

:: Change to the target directory
cd /d "%INSTALL_DIR%"

:: Run the application
echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Exiting...
    exit /b %errorlevel%
)

echo Running application...
python app.py
