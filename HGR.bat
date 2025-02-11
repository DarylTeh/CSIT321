@echo off
setlocal

:: Set the installation directory
set "INSTALL_DIR=%USERPROFILE%\Downloads\HGR"

:: Create the folder if it doesn’t exist
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

:: Extract the ZIP file (assumes a ZIP file named "package.zip" is in the same directory as the BAT)
powershell -Command "Expand-Archive -Path '%~dp0package.zip' -DestinationPath '%INSTALL_DIR%' -Force"

:: Change to the target directory
cd /d "%INSTALL_DIR%"

:: Install dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Exiting...
    exit /b %errorlevel%
)

:: Run the application
echo Running application...
python app.py
