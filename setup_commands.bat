@echo off
REM Setup commands for AdsTable
REM Current Date: 2025-06-18 07:46:34 UTC

echo =====================================================
echo Norwegian Service Providers System - File Setup
echo User: AdsTable
echo Date: 2025-06-18 07:46:34 UTC
echo =====================================================

REM Step 1: Navigate to project directory
cd norwegian-service-providers

REM Step 2: Create your provider_links.py file
echo Creating provider_links.py in src/data/
copy con src\data\provider_links.py

REM Step 3: Create the core modules (copy the code above into these files)
echo.
echo ✅ NOW COPY THE CODE INTO THESE FILES:
echo.
echo 📁 src\core\extraction_engine.py     (Copy the ExtractionEngine code)
echo 📁 src\core\database_manager.py      (Copy the DatabaseManager code) 
echo 📁 src\core\service_explorer.py      (Copy the ServicePlansExplorer code)
echo 📁 database\migrations\002_add_extraction_method.sql (Copy the SQL migration)
echo.

REM Step 4: Setup Python environment
echo Setting up Python environment...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt

REM Step 5: Apply database migration
echo Applying database migration...
sqlite3 database\service_plans.db < database\migrations\002_add_extraction_method.sql

REM Step 6: Test the setup
echo Testing the setup...
python -c "from src.core.service_explorer import main; print('✅ Setup successful!')"

echo.
echo =====================================================
echo ✅ SETUP COMPLETED SUCCESSFULLY!
echo =====================================================
echo.
echo Your files should be placed as follows:
echo norwegian-service-providers\
echo ├── src\
echo │   ├── data\
echo │   │   └── provider_links.py            ⬅️ Your provider configuration
echo │   └── core\
echo │       ├── extraction_engine.py         ⬅️ ExtractionEngine class
echo │       ├── database_manager.py          ⬅️ DatabaseManager class
echo │       └── service_explorer.py          ⬅️ ServicePlansExplorer class
echo └── database\
echo     └── migrations\
echo         └── 002_add_extraction_method.sql ⬅️ Your SQL migration
echo.
echo Next: Run 'python -m src.core.service_explorer' to start the system!
echo.
pause