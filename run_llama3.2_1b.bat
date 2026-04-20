@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set CONFIG_FILE=config_llama3.2_1b_instruct.yaml
set PROMPT_FILE=sample_prompt_introduce.txt
set MAX_RESPONSE=128
set PREFORMATTER=Llama3NoInput

:: Push yaml config file
adb push %CONFIG_FILE% %PHONE_PATH%
adb push %PROMPT_FILE% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main"

:: Run using the below commands
adb shell "cd %PHONE_PATH%; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./main %CONFIG_FILE% -i %PROMPT_FILE% -m %MAX_RESPONSE% --preformatter %PREFORMATTER% --one-prompt-per-line"

:error