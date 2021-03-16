@ECHO OFF 

for /f "tokens=3" %%A in ('find /c /i "Error:" %1') do set "StringErr=%%A"
for /f "tokens=3" %%A in ('find /c /i "Warning:" %1') do set "StringWrn=%%A"
for /f "tokens=3" %%A in ('find /c "Overfull" %1') do set "StringOvf=%%A"
for /f "tokens=3" %%A in ('find /c "Underfull" %1') do set "StringUnf=%%A"


echo Errors:_%StringErr%_   Warnings:_%StringWrn%_   Overfull:_%StringOvf%_   Underfull:_%StringUnf%_ 1>&2

:: return with error code, in order to cause TexMaker to continue to show the message window
:: for this to work, put the following into user defined fast translation box (without the :: ):
:: pdflatex -synctex=1 -interaction=nonstopmode %.tex|!/CountErrWrnBBx.bat %.log|echo "foobar"
::exit /b -1
