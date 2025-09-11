
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import subprocess
import sys
import nltk
import json
import csv

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

# The 'ifeval' task also requires 'punkt_tab'.
try:
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK 'punkt_tab' tokenizer is already downloaded.")
except LookupError:
    print("Downloading NLTK 'punkt_tab' tokenizer data...")
    nltk.download('punkt_tab')
    print("Download complete.")


# --- Configuration ---
BASE_MODEL_DIR = "/root/lanyun-tmp/Falcon/prepared_models"
RESULTS_DIR = "/root/lanyun-tmp/Falcon/falcon_eval_results"
DEVICE = "cuda:0"
BATCH_SIZE = "32"


MODELS_TO_EVAL = [
    "Falcon-E-1B-BitNet",
    "Falcon-E-1B-bf16",
    "Falcon-E-3B-BitNet",
    "Falcon-E-3B-bf16",
]

TASKS = [
    "ifeval",                                # Instruction following
    "arc_challenge",                         # 科学推理问答 
    "gsm8k",                                 # 小学数学应用题 
    "mmlu",                                  # Multi-task language understanding
    "bigbench_strategyqa_generate_until",    # 多步推理 (替换 multiple_choice 版本)
    "truthfulqa_mc2"                         # GPQA 平替 (truthfulqa_mc 的一个有效替代)
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

            # If the output file already exists, remove it to prevent FileExistsError
            if os.path.exists(output_path):
                print(f"Output file {output_path} already exists. Removing it.")
                os.remove(output_path)

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

def summarize_results_to_csv(results_dir):
    """
    Walks through the results directory, finds all JSON results, and creates a CSV summary for each model.
    """
    print("\n--- Starting result summarization... ---")
    for model_name in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        model_results = []
        for result_file in os.listdir(model_dir):
            if result_file.startswith("results_") and result_file.endswith(".json"):
                task_name = result_file.replace("results_", "").replace(".json", "")
                json_path = os.path.join(model_dir, result_file)

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Could not read or parse {json_path}: {e}")
                    continue
                
                results = data.get('results')
                if not results:
                    continue

                for sub_task, metrics in results.items():
                    processed_metrics = {}
                    for metric_name, value in metrics.items():
                        if "_stderr" not in metric_name and isinstance(value, (int, float)):
                            processed_metrics[metric_name] = {'value': value, 'stderr': 'N/A'}
                    
                    for metric_name, value in metrics.items():
                        if "_stderr" in metric_name:
                            base_metric = metric_name.replace("_stderr", "")
                            if base_metric in processed_metrics:
                                processed_metrics[base_metric]['stderr'] = value

                    for metric, values in processed_metrics.items():
                        model_results.append({
                            "task": task_name,
                            "sub_task": sub_task,
                            "metric": metric,
                            "value": values['value'],
                            "stderr": values['stderr']
                        })
        
        if not model_results:
            print(f"No results found to summarize for model: {model_name}")
            continue

        # Sort results for consistency
        model_results.sort(key=lambda x: (x['task'], x['sub_task'], x['metric']))

        summary_csv_path = os.path.join(model_dir, f"{model_name}_summary.csv")
        try:
            with open(summary_csv_path, 'w', newline='') as f_csv:
                fieldnames = ["task", "sub_task", "metric", "value", "stderr"]
                writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(model_results)
            print(f"--- Summary for model {model_name} saved to {summary_csv_path} ---")
        except Exception as e:
            print(f"--- Error writing CSV for model {model_name}: {e} ---")

    print("--- All summaries generated! ---")


if __name__ == "__main__":
    run_evaluation()
    summarize_results_to_csv(RESULTS_DIR)
