import subprocess
import time
import os
import shutil
import sys

# Import functions directly
from train import centralized_train, centralized_test, federated_test
from federated.server import main as server_main
from federated.client import main as client_main

def run_command_in_new_window(command, wait=True):
    process = subprocess.Popen(f'start cmd /k {command}', shell=True)
    
    if wait:
        process.wait()
    return process

def get_input(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default

def main():
    # Interactive prompts
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
        visualize_limit = int(get_input("How many test samples to visualize?", 5))

    # Shared parameters
    if run_federated_train or run_centralized_train or run_federated_test or run_centralized_test:
        num_scenarios = int(get_input("Number of scenarios per client/test", 5))
        batch_size = int(get_input("Batch size", 1))

    # --- Federated Learning --- #
    if run_federated_train:
        print("[INFO] Starting federated learning simulation...")
        server_cmd = f"python -m federated.server --rounds {num_rounds} --client_epochs {client_epochs}"
        server_process = run_command_in_new_window(server_cmd, wait=False)
        print("[INFO] Server started. Waiting for clients...")
        time.sleep(20) # Give server more time to start
        client_processes = []
        for i in range(num_clients):
            client_cmd = (
                f"python -m federated.client --train_dir {train_dir} --val_dir {val_dir} "
                f"--num_scenarios {num_scenarios} --batch_size {batch_size} "
                f"--client_id {i} --num_clients {num_clients}"
            )
            client_processes.append(run_command_in_new_window(client_cmd, wait=False))
            print(f"[INFO] Client {i} started.")
            time.sleep(10) # Give clients more time to connect

        # Wait for federated training to complete (server process will exit when rounds are done)
        server_process.wait()
        for client in client_processes:
            client.wait() # Ensure clients also finish
        print("[INFO] Federated training finished.")

    # --- Centralized Training --- #
    if run_centralized_train:
        print("\n[INFO] Starting centralized training for comparison...")
        centralized_train(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            num_scenarios=num_scenarios,
            epochs=epochs,
            lr=lr
        )
        print("[INFO] Centralized training finished.")

    # --- Testing and Visualization --- #
    if run_centralized_test:
        print("\n[INFO] Running centralized model testing and visualization...")
        centralized_test(
            test_dir=test_dir,
            batch_size=batch_size,
            num_scenarios=num_scenarios,
            visualize_limit=visualize_limit
        )
        print("[INFO] Centralized model testing and visualization complete.")

    if run_federated_test:
        print("\n[INFO] Running federated model testing and visualization...")
        federated_test(
            test_dir=test_dir,
            batch_size=batch_size,
            num_scenarios=num_scenarios,
            visualize_limit=visualize_limit
        )
        print("[INFO] Federated model testing and visualization complete.")

    print("\n[INFO] All processes complete. Open web/index.html in your browser to view results.")

if __name__ == "__main__":
    main()
