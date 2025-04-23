import subprocess
import os
import multiprocessing
from functools import partial

def run_commands_for_idx(scene_name, base_source_path, base_output_path, n_views, iteration, idx):
    """
    Runs the train, render, and eval commands sequentially for a specific index.

    Args:
        scene_name (str): The name of the scene (e.g., 'flower').
        base_source_path (str): Base path for source data.
        base_output_path (str): Base path for output models.
        n_views (int): Number of views used for training.
        iteration (int): Iteration number for rendering and evaluation.
        idx (int): The index for this run (0-9).
    """
    source_path = os.path.join(base_source_path, scene_name)
    # --- Construct model_path based on convention ---
    # Assumes model_path format like: output/{scene}/{scene}_{n_views}_{idx}
    model_path = os.path.join(base_output_path, scene_name, f"{scene_name}_{n_views}_{idx}")
    log_file_path = os.path.join(base_output_path, scene_name, f"run_{scene_name}_{n_views}_{idx}.log")

    print(f"--- Starting run for index {idx}, scene {scene_name} ---")
    print(f"Source Path: {source_path}")
    print(f"Model Path: {model_path}")
    print(f"Log Path: {log_file_path}")

    # # Ensure output directory for the specific run exists
    # os.makedirs(os.path.dirname(model_path), exist_ok=True) # Create output/{scene}/ if needed
    # # Ensure model_path directory itself exists (train.py might do this, but good practice)
    # os.makedirs(model_path, exist_ok=True)

    # Define commands
    # Note: Using f-strings for command construction. Ensure paths with spaces are handled
    # if necessary (e.g., by quoting, though paths here seem okay).
    train_cmd = [
        "python", "train.py",
        "--source_path", source_path,
        "--model_path", model_path,
        "--eval",
        "--n_views", str(n_views),
        "--sample_pseudo_interval", "1",
        "--idx", str(idx) # Assuming train.py doesn't actually need idx directly if model_path is unique
    ]
    render_cmd = [
        "python", "render.py",
        "--source_path", source_path,
        "--model_path", model_path,
        "--iteration", str(iteration),
        "--idx", str(idx) # Assuming render.py doesn't need idx if model_path is unique
    ]
    eval_cmd = [
        "python", "metrics.py",
        # metrics.py seems to take multiple source/model paths, adjust if needed
        # Assuming it evaluates the single model_path specified
        "--source_paths", source_path, # Note: Argument name is plural in metrics.py
        "--model_paths", model_path, # Note: Argument name is plural in metrics.py
        "--iteration", str(iteration),
        "--idx", str(idx) # Assuming metrics.py doesn't need idx if model_path is unique
    ]

    # Execute commands sequentially, logging output
    try:
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Executing: {' '.join(train_cmd)}\n\n")
            log_file.flush()
            print(f"Running Train for idx={idx}...")
            # Using shell=False (recommended), passing args as a list
            result_train = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
            log_file.write("--- Train Output ---\n")
            log_file.write(result_train.stdout)
            log_file.write("\n--- Train Error ---\n")
            log_file.write(result_train.stderr)
            log_file.write("\n--- Train Complete ---\n\n")
            log_file.flush()
            print(f"Train Complete for idx={idx}.")

            log_file.write(f"Executing: {' '.join(render_cmd)}\n\n")
            log_file.flush()
            print(f"Running Render for idx={idx}...")
            result_render = subprocess.run(render_cmd, check=True, capture_output=True, text=True)
            log_file.write("--- Render Output ---\n")
            log_file.write(result_render.stdout)
            log_file.write("\n--- Render Error ---\n")
            log_file.write(result_render.stderr)
            log_file.write("\n--- Render Complete ---\n\n")
            log_file.flush()
            print(f"Render Complete for idx={idx}.")

            log_file.write(f"Executing: {' '.join(eval_cmd)}\n\n")
            log_file.flush()
            print(f"Running Eval for idx={idx}...")
            result_eval = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            log_file.write("--- Eval Output ---\n")
            log_file.write(result_eval.stdout)
            log_file.write("\n--- Eval Error ---\n")
            log_file.write(result_eval.stderr)
            log_file.write("\n--- Eval Complete ---\n\n")
            log_file.flush()
            print(f"Eval Complete for idx={idx}.")

    except subprocess.CalledProcessError as e:
        print(f"!!! Error during execution for index {idx} !!!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Output:\n{e.output}")
        print(f"Stderr:\n{e.stderr}")
        # Log the error to the file as well
        with open(log_file_path, 'a') as log_file:
             log_file.write("\n\n!!! ERROR !!!\n")
             log_file.write(f"Command: {' '.join(e.cmd)}\n")
             log_file.write(f"Return Code: {e.returncode}\n")
             log_file.write(f"Output:\n{e.output}\n")
             log_file.write(f"Stderr:\n{e.stderr}\n")
        # Optionally re-raise the exception or return an error status
        # raise e
        return f"Error in idx {idx}"
    except Exception as e:
        print(f"!!! An unexpected error occurred for index {idx}: {e} !!!")
        with open(log_file_path, 'a') as log_file:
             log_file.write(f"\n\n!!! UNEXPECTED ERROR: {e} !!!\n")
        return f"Unexpected Error in idx {idx}"

    print(f"--- Finished run for index {idx} ---")
    return f"Success for idx {idx}"


if __name__ == "__main__":
    # --- Configuration ---
    scenes = ['fern', 'flower', 'fortress',  'horns',  'leaves',  'orchids',  'room',  'trex'] # Scenes to process
    base_source_path = 'dataset/nerf_llff_data_custom'
    base_output_path = 'output'
    n_views = 3
    iteration = 10000
    num_runs = 10 # Corresponds to idx 0 through 9
    # Adjust number of processes based on your system's CPU cores and GPU availability
    # If jobs are GPU-intensive, you might be limited by the number of GPUs.
    num_processes = num_runs
    
    for scene_name in scenes:

      print(f"Starting parallel execution for scene '{scene_name}' with {num_runs} runs...")
      print(f"Using {num_processes} parallel processes.")

      # Create a partial function with fixed arguments except for 'idx'
      # This is needed for pool.map which only iterates over one argument
      worker_func = partial(run_commands_for_idx, scene_name, base_source_path, base_output_path, n_views, iteration)

      # Create a list of indices to run
      indices_to_run = list(range(num_runs))

      # Create a process pool and run the tasks
      with multiprocessing.Pool(processes=num_processes) as pool:
          results = pool.map(worker_func, indices_to_run)

      print("\n--- Parallel Execution Summary ---")
      for result in results:
          print(result)

      print("\nAll runs completed.")