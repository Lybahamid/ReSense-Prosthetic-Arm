@echo off
echo ===============================
echo   Setting up Grasping Futures project structure
echo ===============================

REM --- Create folders ---
mkdir Data\DB5
mkdir Notebooks
mkdir Src
mkdir Outputs
mkdir Outputs\models
mkdir Outputs\logs
mkdir Outputs\figures

REM --- Create empty files ---
type nul > Notebooks\db5_preprocessing.ipynb
type nul > Src\__init__.py
type nul > Src\data_loader.py
type nul > Src\preprocessing.py
type nul > Src\models.py
type nul > Src\train.py
type nul > Src\utils.py
type nul > check_libs.py
type nul > requirements.txt
type nul > README.md

echo ===============================
echo   Project structure created successfully!
echo ===============================
pause
