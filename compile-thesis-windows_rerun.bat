SET FILENAME=thesis
rem MIKTEX_BIN=C:\miktex\texmfs\install\miktex\bin\
rem set PATH=%MIKTEX_BIN%;%PATH%


rem bibtex "%FILENAME%.aux"
rem makeglossaries "%FILENAME%"
pdflatex -shell-escape  -synctex=1 -interaction=nonstopmode -interaction=nonstopmode -extra-mem-top=50000000  -extra-mem-bot=10000000  -main-memory=90000000 "%FILENAME%.tex"

countErrWrnBBx.bat "%FILENAME%.log"



#"%FILENAME%.pdf"
