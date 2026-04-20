@echo off

set TEMP_FILELIST_PATH=clang_format_list_%RANDOM%.txt"

:: Create empty temporary file
break > %TEMP_FILELIST_PATH%

:: Append each *.cpp and *.h files to be formatted into the temporary file
for /f "delims=" %%A in ('dir /S/B *.cpp *.h ^| findstr /V /C:"third_party" /C:"backend\api"') do (
    echo %%A >> %TEMP_FILELIST_PATH%
)

:: Begin formatting the source files
clang-format -i --verbose --files=%TEMP_FILELIST_PATH%

:: Delete the temporary file
del %TEMP_FILELIST_PATH%