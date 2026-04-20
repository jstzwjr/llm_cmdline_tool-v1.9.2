@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set CONFIG_FILE=config_llava.yaml
set PREFORMATTER=VicunaNoInput

:: Push yaml config file
adb push %CONFIG_FILE% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main_llava"

:: Run using the below commands
adb shell "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%PHONE_PATH% %PHONE_PATH%/main_llava %PHONE_PATH%/%CONFIG_FILE%"

:error