import subprocess
import time
import os
import shutil
import sys

# Import functions directly
from train import centralized_train, centralized_test, federated_test
# The server and client main functions are called via subprocess, so direct import is not strictly needed
# from federated.server import main as server_main
# from federated.client import main as client_main

def wait_for_completion(completion_file, timeout=1800, check_interval=10):
    """
    Wait for a completion file to be created with 'completed' content
    
    Args:
        completion_file: Path to the completion file to watch
        timeout: Maximum time to wait in seconds (default: 30 minutes)
        check_interval: How often to check for the file in seconds (default: 10 seconds)
    
    Returns:
        bool: True if completed successfully, False if timed out
    """
    start_time = time.time()
    print(f"[INFO] Waiting for completion marker: {completion_file}")
    
    while time.time() - start_time < timeout:
        if os.path.exists(completion_file):
            try:
                with open(completion_file, 'r') as f:
                    content = f.read().strip()
                if content == "completed":
                    print(f"[INFO] ✅ Completion marker found: {completion_file}")
                    # Clean up the completion file
                    os.remove(completion_file)
                    return True
            except Exception as e:
                print(f"[WARNING] Error reading completion file: {e}")
        
        print(f"[INFO] Waiting... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(check_interval)
    
    print(f"[WARNING] Timeout waiting for completion marker after {timeout} seconds")
    return False

def run_command_in_new_window(command, wait=True):
    if wait:
        # Run directly in current process for waiting
        process = subprocess.Popen(command, shell=True)
        process.wait()
        return process
    else:
        # Run in new window for non-waiting processes
        process = subprocess.Popen(f'start cmd /k {command}', shell=True)
        return process

def get_input(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default

def main():
    # Interactive prompts
    model_choice = get_input("Select model: [1] GATv2, [2] VectorNet", "1")
    model_name = "GATv2" if model_choice == "1" else "VectorNet"

    run_mode = get_input("Select mode: [1] Federated, [2] Centralized, [3] Both", "1")

    # Federated options
    if run_mode in ["1", "3"]:
        fed_action = get_input("Federated: [1] Train, [2] Test, [3] Both", "1")
        run_federated_train = fed_action in ["1", "3"]
        run_federated_test = fed_action in ["2", "3"]
    else:
        run_federated_train, run_federated_test = False, False

    # Centralized options
    if run_mode in ["2", "3"]:
        cen_action = get_input("Centralized: [1] Train, [2] Test, [3] Both", "1")
        run_centralized_train = cen_action in ["1", "3"]
        run_centralized_test = cen_action in ["2", "3"]
    else:
        run_centralized_train, run_centralized_test = False, False

    # Parameters for training
    if run_federated_train or run_centralized_train:
        train_dir = get_input("Train directory", "dataset/train_small")
        val_dir = get_input("Validation directory", "dataset/val_small")
        lr = float(get_input("Learning rate", 1e-3))

    if run_federated_train:
        num_rounds = int(get_input("Number of federated rounds", 3))
        num_clients = int(get_input("Number of clients", 2))
        client_epochs = int(get_input("Number of epochs for client training", 1))

    if run_centralized_train:
        epochs = int(get_input("Epochs for centralized training", 2))

    # Parameters for testing
    if run_federated_test or run_centralized_test:
        test_dir = get_input("Test directory", "dataset/test_small")
        visualize_limit = int(get_input("How many test samples to visualize?", 2))

    # Shared parameters
    if run_federated_train or run_centralized_train or run_federated_test or run_centralized_test:
        num_scenarios = int(get_input("Number of scenarios per client/test", 5))
        batch_size = int(get_input("Batch size", 1))
        seq_len = int(get_input("Sequence length for trajectory prediction", 30))

    # --- Federated Learning --- #
    if run_federated_train:
        print("[INFO] Starting federated learning simulation...")
        
        # Create completion file path
        completion_file = os.path.join("temp", "federated_training_completed.txt")
        
        # Remove any existing completion file
        if os.path.exists(completion_file):
            os.remove(completion_file)
        
        # Start server in new window
        server_cmd = f"python -m federated.server --rounds {num_rounds} --client_epochs {client_epochs} --model_name {model_name} --seq_len {seq_len}"
        server_process = subprocess.Popen(f'start cmd /k {server_cmd}', shell=True)
        print("[INFO] Server started in new window. Waiting for clients...")
        time.sleep(20) # Give server more time to start
        
        # Start clients in new windows
        client_processes = []
        for i in range(num_clients):
            client_cmd = (
                f"python -m federated.client --train_dir {train_dir} --val_dir {val_dir} "
                f"--num_scenarios {num_scenarios} --batch_size {batch_size} --seq_len {seq_len} "
                f"--client_id {i} --num_clients {num_clients} --model_name {model_name}"
            )
            client_process = subprocess.Popen(f'start cmd /k {client_cmd}', shell=True)
            client_processes.append(client_process)
            print(f"[INFO] Client {i} started in new window.")
            time.sleep(10) # Give clients more time to connect

        # Wait for federated training to complete using completion file
        print("[INFO] Waiting for federated training to complete...")
        print("[INFO] Check the opened console windows for training progress...")
        
        # Wait for completion marker
        if wait_for_completion(completion_file, timeout=1800):  # 30 minutes timeout
            print("[INFO] ✅ Federated training completed successfully!")
        else:
            print("[WARNING] Federated training may not have completed properly")
        
        print("[INFO] Federated training finished.")

    # --- Centralized Training --- #
    if run_centralized_train:
        print("\n[INFO] Starting centralized training for comparison...")
        centralized_train(
            model_name=model_name,
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            num_scenarios=num_scenarios,
            epochs=epochs,
            lr=lr,
            seq_len=seq_len
        )
        print("[INFO] Centralized training finished.")

    # --- Testing and Visualization --- #
    if run_centralized_test:
        print("\n[INFO] Running centralized model testing and visualization...")
        centralized_test(
            model_name=model_name,
            test_dir=test_dir,
            batch_size=batch_size,
            num_scenarios=num_scenarios,
            visualize_limit=visualize_limit,
            seq_len=seq_len
        )
        print("[INFO] Centralized model testing and visualization complete.")

    if run_federated_test:
        print("\n[INFO] Running federated model testing and visualization...")
        federated_test(
            model_name=model_name,
            test_dir=test_dir,
            batch_size=batch_size,
            num_scenarios=num_scenarios,
            visualize_limit=visualize_limit,
            seq_len=seq_len
        )
        print("[INFO] Federated model testing and visualization complete.")

    print("\n[INFO] All processes complete. Open web/index.html in your browser to view results.")

if __name__ == "__main__":
    main()