@echo on
cd /d "%~dp0"
echo Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Exiting...
    exit /b %errorlevel%
)
echo Running application...
python app.py