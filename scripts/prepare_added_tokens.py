import os
import json
import yaml
import argparse


def export_from_tokenizer_json(json_config, output_dir, binary=False):
    if 'added_tokens' not in json_config:
        print("Added token is not found in tokenizer.json")
        return

    added_tokens_config = json_config['added_tokens']

    # Convert key to utf-8 and flip key and value
    added_tokens_map = {}
    for added_token in added_tokens_config:
        tokenId = added_token['id']
        tokenStr = added_token['content']
        added_tokens_map[tokenId] = tokenStr.encode('utf-8') if binary else tokenStr

    with open(os.path.join(output_dir, 'added_tokens.yaml'), 'w') as outfile:
        yaml.dump(added_tokens_map, outfile)
    print("Exported 'added_tokens.yaml' from 'tokenizer.json'")


def export_from_added_tokens_json(json_config, output_dir, binary=False):
    # Convert key to utf-8 and flip key and value
    added_tokens_map = {}
    for tokenStr, tokenId in json_config.items():
        added_tokens_map[tokenId] = tokenStr.encode('utf-8') if binary else tokenStr

    with open(os.path.join(output_dir, 'added_tokens.yaml'), 'w') as outfile:
        yaml.dump(added_tokens_map, outfile)
    print("Exported 'added_tokens.yaml' from 'added_tokens.json'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config_path', help="The path to either 'tokenizer.json' or 'added_tokens.json'.")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help="The output directory path where added_tokens.yaml will be exported. Default to the directory path of the input json file.")
    parser.add_argument('-b', '--binary', action='store_true', help="Export the added tokens as base64 encoded binary format instead of string literal. Default to exporting in string literal.")
    args = parser.parse_args()

    json_config_path = args.json_config_path

    added_tokens_processor_map = {
        'tokenizer.json': export_from_tokenizer_json,
        'added_tokens.json': export_from_added_tokens_json
    }
    filename = os.path.basename(json_config_path).lower()
    added_tokens_processor = added_tokens_processor_map[filename]

    with open(json_config_path, 'rb') as f:
        json_config = json.load(f)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(json_config_path)

    added_tokens_processor(json_config, output_dir, args.binary)
