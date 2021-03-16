SETLOCAL EnableExtensions EnableDelayedExpansion
SET FILENAME=thesis
rem MIKTEX_BIN=C:\miktex\texmfs\install\miktex\bin\
rem set PATH=%MIKTEX_BIN%;%PATH%
set ROOT_DIR=%~dp0
cd %ROOT_DIR%
DEL /Q "%FILENAME%.aux"  >nul 2>&1
DEL /Q "%FILENAME%.bbl"  >nul 2>&1
DEL /Q "%FILENAME%.blg"  >nul 2>&1
DEL /Q "%FILENAME%.d" >nul 2>&1
DEL /Q "%FILENAME%.fls" >nul 2>&1
DEL /Q "%FILENAME%.ild" >nul 2>&1
DEL /Q "%FILENAME%.ind" >nul 2>&1
DEL /Q "%FILENAME%.toc" >nul 2>&1
DEL /Q "%FILENAME%.lot" >nul 2>&1
DEL /Q "%FILENAME%.lof" >nul 2>&1
DEL /Q "%FILENAME%.idx" >nul 2>&1
DEL /Q "%FILENAME%.out" >nul 2>&1
DEL /Q "%FILENAME%.nlo" >nul 2>&1
DEL /Q "%FILENAME%.nls" >nul 2>&1
DEL /Q "%FILENAME%.pdf" >nul 2>&1
DEL /Q "%FILENAME%.ps" >nul 2>&1
DEL /Q "%FILENAME%.dvi" >nul 2>&1
DEL /Q "%FILENAME%.bcf" >nul 2>&1
DEL /Q "%FILENAME%.aux.bbl" >nul 2>&1
DEL /Q "%FILENAME%.aux.blg" >nul 2>&1

DEL /Q "%FILENAME%.aux" >nul 2>&1
DEL /Q "%FILENAME%.acn" >nul 2>&1
DEL /Q "%FILENAME%.acr" >nul 2>&1
DEL /Q "%FILENAME%.alg" >nul 2>&1
DEL /Q "%FILENAME%.auxlock" >nul 2>&1
DEL /Q "%FILENAME%.bbl" >nul 2>&1
DEL /Q "%FILENAME%.blg" >nul 2>&1
DEL /Q "%FILENAME%.d" >nul 2>&1
DEL /Q "%FILENAME%.fls" >nul 2>&1
DEL /Q "%FILENAME%.ild" >nul 2>&1
DEL /Q "%FILENAME%.ilg" >nul 2>&1
DEL /Q "%FILENAME%.ind" >nul 2>&1
DEL /Q "%FILENAME%.ist" >nul 2>&1
DEL /Q "%FILENAME%.toc" >nul 2>&1
DEL /Q "%FILENAME%.lot" >nul 2>&1
DEL /Q "%FILENAME%.lof" >nul 2>&1
DEL /Q "%FILENAME%.idx" >nul 2>&1
DEL /Q "%FILENAME%.out" >nul 2>&1
DEL /Q "%FILENAME%.pre" >nul 2>&1
DEL /Q "%FILENAME%.nlo" >nul 2>&1
DEL /Q "%FILENAME%.nls" >nul 2>&1
DEL /Q "%FILENAME%.slg" >nul 2>&1
DEL /Q "%FILENAME%.syg" >nul 2>&1
DEL /Q "%FILENAME%.syi" >nul 2>&1
DEL /Q "%FILENAME%.bcf" >nul 2>&1
DEL /Q "%FILENAME%.aux.bbl" >nul 2>&1
DEL /Q "%FILENAME%.aux.blg" >nul 2>&1

cd %ROOT_DIR%


lualatex --shell-escape  --draftmode --synctex=1 --interaction=nonstopmode  "%FILENAME%.tex"
makeglossaries "%FILENAME%"
rem bibtex "%FILENAME%.aux"
biber "%FILENAME%"
lualatex --shell-escape --draftmode --synctex=1  --interaction=nonstopmode  "%FILENAME%.tex"
makeglossaries "%FILENAME%"
lualatex --shell-escape  --synctex=1  --interaction=nonstopmode "%FILENAME%.tex"

countErrWrnBBx.bat "%FILENAME%.log"




