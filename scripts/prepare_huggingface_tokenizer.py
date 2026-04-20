import os
import json
import argparse
import prepare_added_tokens

def export_vocab_txt(json_config, output_dir):
    vocab_txt_path = os.path.join(output_dir, 'vocab.txt')
    open(vocab_txt_path, 'w').close() # Clear file content if file existed
    with open(vocab_txt_path, 'ab') as f:
        for token_str, token_id in json_config['model']['vocab'].items():
            f.write(token_str.encode('utf8'))
            f.write(f"\n{token_id}\n".encode('utf8'))
    print("Exported 'vocab.txt' from 'tokenizer.json'")


def export_merges_txt(json_config, output_dir):
    if 'merges' not in json_config['model']:
        return
    merges_txt_path = os.path.join(output_dir, 'merges.txt')
    open(merges_txt_path, 'w').close() # Clear file content if file existed
    with open(merges_txt_path, 'ab') as f:
        f.write("#version\n".encode('utf8')) # Write version comment header
        for line in json_config['model']['merges']:
            if (isinstance(line, list)):
                line = ' '.join(line)
            f.write(line.encode('utf8') + b'\n')
    print("Exported 'merges.txt' from 'tokenizer.json'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config_path', help="The path to 'tokenizer.json'.")
    parser.add_argument('-o', '--output_dir', type=str, default=None, help="The output directory path where the tokenizer files ('vocab.txt' and 'added_tokens.yaml') will be exported. Default to the directory path of the input json file.")
    parser.add_argument('-b', '--binary', action='store_true', help="Export the added tokens as base64 encoded binary format instead of string literal. Default to exporting in string literal.")
    args = parser.parse_args()

    json_config_path = args.json_config_path

    filename = os.path.basename(json_config_path).lower()

    with open(json_config_path, 'rb') as f:
        json_config = json.load(f)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(json_config_path)

    # Export vocab.txt
    export_vocab_txt(json_config, output_dir)

    # Export merges.txt
    export_merges_txt(json_config, output_dir)

    # Export added_tokens.yaml
    prepare_added_tokens.export_from_tokenizer_json(json_config, output_dir, args.binary)
