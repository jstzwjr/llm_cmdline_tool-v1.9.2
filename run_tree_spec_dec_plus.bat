@echo on

:: Setup phone environment
set PHONE_PATH=/data/local/tmp/llm_sdk

set TARGET_CONFIG_FILE=config_vicuna_7b_spec_dec_target_8t_i8c_2048c.yaml
set DRAFT_CONFIG_FILE=config_vicuna_160m_spec_dec_draft_4t_i8c_2048c.yaml
set INPUT_PROMPT=mt_bench_1st_turn.txt
set MAX_RESPONSE=256
set PREFORMATTER=VicunaNoInput
set TARGET_TEMPERATURE=0.0
set DRAFT_TEMPERATURE=0.0

:: Push yaml config file
adb push %INPUT_PROMPT% %PHONE_PATH%
adb push %TARGET_CONFIG_FILE% %PHONE_PATH%
adb push %DRAFT_CONFIG_FILE% %PHONE_PATH%

:: Set execute permission
adb shell "chmod +x %PHONE_PATH%/main_tree_spec_dec_plus"

:: Run using the below commands
set TOP_K=4,2,1
set INFER_TYPE=0
set DRAFT_LENGTH=3
set DRAFT_LENGTH_AUX=7
set MAX_TOL=5
set MIN_TOL=2
adb shell "cd %PHONE_PATH%; LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD ./main_tree_spec_dec_plus %TARGET_CONFIG_FILE% --infer-type %INFER_TYPE% --draft %DRAFT_CONFIG_FILE% --draft-len %DRAFT_LENGTH% -i %INPUT_PROMPT% -m %MAX_RESPONSE% --preformatter %PREFORMATTER% --one-prompt-per-line --target-temperature %TARGET_TEMPERATURE% --draft-temperature %DRAFT_TEMPERATURE% --draft-len-aux %DRAFT_LENGTH_AUX% --max-tol %MAX_TOL% --min-tol %MIN_TOL%  --tree-topk %TOP_K%"

:error