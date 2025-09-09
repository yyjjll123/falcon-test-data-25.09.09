
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import subprocess
import sys
import nltk

# --- Download necessary NLTK data ---
# The 'ifeval' task requires the 'punkt' tokenizer from NLTK.
# This ensures it's downloaded before the evaluation starts.
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer is already downloaded.")
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer data...")
    nltk.download('punkt')
    print("Download complete.")


# --- Configuration ---
BASE_MODEL_DIR = "/root/lanyun-tmp/prepared_models"
RESULTS_DIR = "/root/lanyun-tmp/falcon_eval_results"
DEVICE = "cuda:0"
BATCH_SIZE = "8"


MODELS_TO_EVAL = [
    "Falcon-E-1B-BitNet",
    "Falcon-E-1B-bf16",
    "Falcon-E-3B-BitNet",
    "Falcon-E-3B-bf16",
]

TASKS = [
    "ifeval",
    "hendrycks_test-math",
    "gpqa",  # Removed due to gated dataset access issues，需要登陆Hugging Face才能使用
    "musr",  # Removed as it's not a valid task name，无效名字，不确定
    "bbh",
    "mmlu_pro",
]

def run_evaluation():
    """
    Iterates through models and tasks, running the lm-evaluation-harness for each combination.
    """
    print("Starting evaluation process...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model_name in MODELS_TO_EVAL:
        model_path = os.path.join(BASE_MODEL_DIR, model_name)
        if not os.path.isdir(model_path):
            print(f"Warning: Model directory not found for {model_name} at {model_path}. Skipping.")
            continue

        for task in TASKS:
            print(f"--- Running evaluation for model: {model_name} on task: {task} ---")

            output_dir_for_model = os.path.join(RESULTS_DIR, model_name)
            os.makedirs(output_dir_for_model, exist_ok=True)
            output_path = os.path.join(output_dir_for_model, f"results_{task}.json")

            command = [
                "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
                "--tasks", task,
                "--device", DEVICE,
                "--batch_size", BATCH_SIZE,
                "--output_path", output_path,
                "--log_samples"
            ]

            print(f"Executing command: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(line, end='')
                    sys.stdout.flush()

            return_code = process.wait()

            if return_code == 0:
                print(f"--- Successfully completed task: {task} for model: {model_name} ---")
            else:
                print(f"--- Error running evaluation for model: {model_name} on task: {task}. Return code: {return_code} ---")
                continue

    print("--- All evaluations complete! ---")
    print(f"Results are saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_evaluation()