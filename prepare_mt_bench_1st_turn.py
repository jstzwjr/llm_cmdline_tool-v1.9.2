# Refer to the following link to access the mt_bench jsonl file.
# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl
# Kindly ensure that the question.jsonl file is downloaded to the same directory as this file.
"""This file can extract needed questions in the jsonl file."""
import json

if __name__ == '__main__':
    JSONL_FILE = 'question.jsonl'

    # Retrieve the 1st turn question of each instance in the jsonl file.
    with open(JSONL_FILE, 'r') as input_file:
        all_instance_list = input_file.readlines()

    for qs_id, qs_list in enumerate(all_instance_list):
        instance = json.loads(qs_list)
        prompt = instance['turns'][0].encode('utf-8')
        all_instance_list[qs_id] = prompt.replace(b'\n', b'\\n')

    # Dump each prompt into mt_bench_1st_turn.txt file
    with open('mt_bench_1st_turn.txt', 'wb') as output_file:
        for prompt in all_instance_list[:-1]:
            output_file.write(prompt)
            output_file.write(b'\n')
        output_file.write(all_instance_list[-1])