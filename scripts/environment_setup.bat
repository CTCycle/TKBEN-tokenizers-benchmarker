@echo off
rem Use this script to create a new environment called "TokenExp"

echo STEP 1: Creation of TokenExp environment
call conda create -n TokenExp python=3.11 -y
if errorlevel 1 (
    echo Failed to create the environment TokenExp
    goto :eof
)

rem If present, activate the environment
call conda activate TokenExp

rem Install additional packages with pip
echo STEP 2: Install python libraries and packages
call pip install numpy==1.26.4 pandas==2.1.4 openpyxl==3.1.5 tqdm==4.66.4
call pip install transformers==4.43.3 sentencepiece==0.2.0 datasets=2.20.0 sacremoses=0.1.1
call pip install seaborn==0.13.2 matplotlib==3.9.0
if errorlevel 1 (
    echo Failed to install Python libraries.
    goto :eof
)

@echo off
rem install packages in editable mode
echo STEP 3: Install utils packages in editable mode
call cd .. && pip install -e . --use-pep517
if errorlevel 1 (
    echo Failed to install the package in editable mode
    goto :eof
)

rem Print the list of dependencies installed in the environment
echo List of installed dependencies
call conda list

set/p<nul =Press any key to exit... & pause>nul



