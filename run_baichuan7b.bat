@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set CONFIG_FILE=config_baichuan7b.yaml

set PROMPT_FILE=sample_prompt.txt

:: Push yaml config file
adb push %CONFIG_FILE% %PHONE_PATH%

:: Push sample prompt file
adb push %PROMPT_FILE% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main"

:: Run using the below commands
adb shell "cd %PHONE_PATH%; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./main %CONFIG_FILE% -i %PROMPT_FILE% -m 100"

:error