@echo off

:: [CHECK CUSTOM ENVIRONMENTS] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Check if TokenExplorer environment is available or use custom environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
call conda config --add channels conda-forge
call conda info --envs | findstr "TokenExplorer"
if %ERRORLEVEL%==0 (
    echo TokenExplorer environment detected
    call conda activate TokenExplorer
    goto :dependencies
) else (
    echo TokenExplorer environment has not been found, it will now be created using python 3.11
    echo Depending on your internet connection, this may take a while!
    call conda create -n TokenExplorer python=3.11 -y
    call conda activate TokenExplorer
    goto :dependencies
)

:: [INSTALL DEPENDENCIES] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install dependencies to python environment
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:dependencies
echo.
echo Install python libraries and packages
call pip install numpy==1.26.4 pandas==2.1.4 openpyxl==3.1.5 tqdm==4.66.4
call pip install transformers==4.43.3 sentencepiece==0.2.0 datasets==2.20.0 sacremoses==0.1.1
call pip install seaborn==0.13.2 matplotlib==3.9.0
call pip install ipykernel==6.29.5

:: [INSTALL PROJECT IN EDITABLE MODE] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Install project in developer mode
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo Install utils packages in editable mode
call cd .. && pip install -e . --use-pep517 && cd TokenExplorer

:: [CLEAN CACHE] 
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Clean packages cache
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
echo.
echo Cleaning conda and pip cache 
call conda clean --all -y
call pip cache purge

:: [SHOW LIST OF INSTALLED DEPENDENCIES]
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Show installed dependencies
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
echo.
echo List of installed dependencies:
call conda list

echo.
echo Installation complete. You can now run TokenExplorer on this system!
pause