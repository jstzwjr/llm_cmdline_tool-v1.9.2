import argparse
import os
import sentencepiece.sentencepiece_model_pb2 as model

def disable_dummy_prefix(tokenizer_model_path, output_dir):
    # Import tokenizer model
    m = model.ModelProto()
    with open(tokenizer_model_path, "rb") as f:
        m.ParseFromString(f.read())

    # Override `add_dummy_prefix` to False
    print("Disabling 'add_dummy_prefix'")
    normalizer_spec = model.NormalizerSpec()
    normalizer_spec.add_dummy_prefix = False
    m.normalizer_spec.MergeFrom(normalizer_spec)

    # Export modified tokenizer model
    input_filename, ext = os.path.splitext(os.path.basename(tokenizer_model_path))
    output_filename = input_filename + "_nodummyprefix" + ext
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "wb") as f:
        f.write(m.SerializeToString())
    print(f"Exported '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SentencePiece tokenizer model file by disabling `add_dummy_prefix` option.")
    parser.add_argument("tokenizer_path", type=str, default="./tokenizer.model", help="The path to tokenizer model file.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The output directory path where the modified tokenizer model will be exported. Default to the directory path of the input tokenizer model file.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.tokenizer_path)

    disable_dummy_prefix(args.tokenizer_path, output_dir)
