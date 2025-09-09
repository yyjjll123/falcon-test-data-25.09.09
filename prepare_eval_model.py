
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from onebitllms import replace_linear_with_bitnet_linear
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare a model for evaluation with lm-evaluation-harness.")
    parser.add_argument("--model_id", type=str, help="Hugging Face model ID (e.g., 'tiiuae/Falcon-E-1B-Base').")
    parser.add_argument("--revision", type=str, help="Model revision to load from Hugging Face Hub (e.g., 'prequantized', 'bfloat16').")
    parser.add_argument("--model_path", type=str, help="Local path to the model directory. Overrides model_id if provided.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the prepared model.")
    parser.add_argument("--bitnet", action="store_true", help="Apply BitNet transformation for prequantized models.")

    args = parser.parse_args()

    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    load_path = args.model_path if args.model_path else args.model_id
    if not load_path:
        raise ValueError("Either --model_path or --model_id must be provided.")

    print(f"Loading tokenizer from {load_path}...")
    # For prequantized models, the tokenizer is usually in the main branch/directory, so we don't specify revision here.
    # For bfloat16, it should be the same.
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        revision=args.revision if not args.model_path else None,
    )

    print(f"Loading model from {load_path} with revision '{args.revision}'...")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        revision=args.revision if not args.model_path else None,
        torch_dtype=torch.bfloat16,
        device_map="auto" # Use GPU if available
    )

    if args.bitnet:
        print("Applying BitNet transformation...")
        # This function expects a model that has been loaded from a 'prequantized' state.
        model = replace_linear_with_bitnet_linear(model)

    print(f"Saving prepared model and tokenizer to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Successfully prepared and saved model to {args.output_dir}")

if __name__ == "__main__":
    # Set HF_ENDPOINT for mirror
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    main()