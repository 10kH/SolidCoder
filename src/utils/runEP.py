"""
EvalPlus runner utility for HumanEval evaluation.
Uses docker to run evalplus in a containerized environment.
"""
import subprocess
import os

def run_eval_plus(RESULTS_PATH, SUMMARY_PATH, DATASET):
    """
    Run EvalPlus evaluation using Docker.
    
    Args:
        RESULTS_PATH: Path to the results JSONL file (relative to project root)
        SUMMARY_PATH: Path where summary will be saved (relative to project root)
        DATASET: Dataset name ('humaneval' or 'mbpp')
    """
    # Get project root directory (assuming this script is in src/utils/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Build docker command with project root mounted
    command = (
        f"docker run -v {project_root}:/app ganler/evalplus:latest "
        f"--dataset {DATASET} --samples /app/{RESULTS_PATH} "
        f"> {os.path.join(project_root, SUMMARY_PATH)}"
    )
    
    print(f"Running EvalPlus: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print("Output:\n", result.stdout)
        if result.stderr:
            print("Error:\n", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error occurred: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) >= 4:
        run_eval_plus(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python runEP.py <results_path> <summary_path> <dataset>")
        print("Example: python runEP.py results/HumanEval/Direct/gpt-4o/Results.jsonl summary.txt humaneval")
