import argparse

def quantize_weights(input_path, output_path):
    print(f"Quantizing {input_path} to {output_path} (1.58-bit ternary format)...")
    # Stub: load HF weights, apply BitNet scaling, pack to 2-bit, interleave, save.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantize_weights(args.input, args.output)
