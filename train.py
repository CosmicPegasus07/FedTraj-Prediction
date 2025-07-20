import argparse
import os
import torch
import json
from datetime import datetime
import numpy as np

from models.gnn import GATv2TrajectoryPredictor
from models.vectornet import VectorNet
from utils.data_utils import get_pyg_data_loader
from utils.viz_utils import visualize_predictions
from utils.loss_functions import ImprovedTrajectoryLoss, compute_trajectory_metrics

def get_model(model_name, in_channels, hidden_channels, out_channels, seq_len=30):
    if model_name == "GATv2":
        return GATv2TrajectoryPredictor(in_channels, hidden_channels, out_channels * seq_len, seq_len=seq_len)
    elif model_name == "VectorNet":
        return VectorNet(in_channels, out_channels * seq_len, hidden_dim=hidden_channels, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def evaluate(model, loader, device, seq_len=30):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    all_displacement_errors = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            outputs = outputs.view(-1, seq_len, 2)
            
            labels = batch.y
            labels = labels.view(-1, seq_len, 2)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred_np = outputs.cpu().numpy()
            gt_np = labels.cpu().numpy()
            displacement_error = np.linalg.norm(pred_np - gt_np, axis=2).mean(axis=1)
            all_displacement_errors.extend(displacement_error.tolist())

    avg_loss = total_loss / len(loader)
    
    if all_displacement_errors:
        min_ade_k1 = np.mean(all_displacement_errors)
        min_fde_k1 = np.mean(all_displacement_errors)
        mr_2m = np.mean(np.array(all_displacement_errors) > 2.0)
    else:
        min_ade_k1 = float('inf')
        min_fde_k1 = float('inf')
        mr_2m = float('inf')

    print(f"[INFO] Validation/Test Loss: {avg_loss:.4f}, minADE(K=1): {min_ade_k1:.4f}, minFDE(K=1): {min_fde_k1:.4f}, MR(2.0m): {mr_2m:.4f}")
    return avg_loss, min_ade_k1, min_fde_k1, mr_2m

def centralized_train(train_dir, val_dir, batch_size=32, num_scenarios=-1, epochs=5, lr=1e-3, model_name="GATv2", seq_len=30):
    print("[INFO] Starting centralized training...")

    # Use more conservative settings for data loading
    train_loader = get_pyg_data_loader(
        train_dir, batch_size, num_scenarios, shuffle=True, mode='train', seq_len=seq_len,
        use_multiprocessing=True  # Try multiprocessing first, will fallback if needed
    )
    val_loader = get_pyg_data_loader(
        val_dir, batch_size, min(num_scenarios, 500) if num_scenarios > 0 else 500,
        shuffle=False, mode='val', seq_len=seq_len,
        use_multiprocessing=True  # Try multiprocessing first, will fallback if needed
    )

    if not train_loader:
        print("[ERROR] Could not create training data loader. Aborting training.")
        return

    if not val_loader:
        print("[WARNING] Could not create validation data loader. Training without validation.")
        val_loader = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, in_channels=5, hidden_channels=32, out_channels=2, seq_len=seq_len).to(device)

    model_path = f"saved_models/centralized_model_{model_name}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[INFO] Using existing model to train further: {model_path}")
    else:
        print("[INFO] Model not found, starting fresh.")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch)
            outputs = outputs.view(-1, seq_len, 2)

            labels = batch.y
            labels = labels.view(-1, seq_len, 2)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss/len(train_loader)
        print(f"[INFO] Epoch {epoch+1} complete. Avg Train Loss: {avg_train_loss:.4f}")

        # Only evaluate on validation set if available
        if val_loader is not None:
            val_loss, val_min_ade, val_min_fde, val_mr = evaluate(model, val_loader, device, seq_len=seq_len)
        else:
            val_loss, val_min_ade, val_min_fde, val_mr = 0.0, 0.0, 0.0, 0.0
            print("[INFO] No validation data available.")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_min_ade_k1': val_min_ade,
            'val_min_fde_k1': val_min_fde,
            'val_mr_2m': val_mr
        })

    torch.save(model.state_dict(), model_path)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "centralized_training_history.json")
    
    all_histories = []
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            try:
                all_histories = json.load(f)
            except json.JSONDecodeError:
                all_histories = []

    run_metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "epochs_trained": epochs,
        "learning_rate": lr,
        "history": history
    }
    all_histories.append(run_metadata)

    with open(history_path, 'w') as f:
        json.dump(all_histories, f, indent=4)

def federated_train_complete(model_name="GATv2", num_clients=2, num_rounds=3, train_dir="dataset/train_small", val_dir="dataset/val_small", num_scenarios=100, batch_size=32, epochs=1, lr=1e-3, seq_len=30):
    """Complete federated training function that runs server and clients in sequence"""
    print(f"[INFO] Starting complete federated training for {model_name}")
    print(f"[INFO] Configuration: {num_clients} clients, {num_rounds} rounds, {epochs} epochs per round")

    try:
        # Import federated components
        import subprocess
        import time
        import threading
        from pathlib import Path

        # Start server in background
        print("[INFO] Starting federated server...")
        server_cmd = [
            "python", "-m", "federated.server",
            "--rounds", str(num_rounds),
            "--client_epochs", str(epochs),
            "--model_name", model_name
        ]

        server_process = subprocess.Popen(
            server_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Wait a bit for server to start
        time.sleep(10)
        print("[INFO] Server started, launching clients...")

        # Start clients
        client_processes = []
        for i in range(num_clients):
            print(f"[INFO] Starting client {i+1}/{num_clients}...")
            client_cmd = [
                "python", "-m", "federated.client",
                "--train_dir", train_dir,
                "--val_dir", val_dir,
                "--num_scenarios", str(num_scenarios),
                "--batch_size", str(batch_size),
                "--client_id", str(i),
                "--num_clients", str(num_clients),
                "--model_name", model_name
            ]

            client_process = subprocess.Popen(
                client_cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            client_processes.append(client_process)
            time.sleep(5)  # Stagger client starts

        print("[INFO] All clients started, waiting for completion...")

        # Monitor server output
        def monitor_server():
            for line in iter(server_process.stdout.readline, ''):
                if line:
                    print(f"[SERVER] {line.strip()}")

        server_thread = threading.Thread(target=monitor_server)
        server_thread.daemon = True
        server_thread.start()

        # Monitor client outputs
        def monitor_client(client_id, process):
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[CLIENT-{client_id}] {line.strip()}")

        client_threads = []
        for i, client_process in enumerate(client_processes):
            thread = threading.Thread(target=monitor_client, args=(i, client_process))
            thread.daemon = True
            thread.start()
            client_threads.append(thread)

        # Wait for server to complete
        server_return_code = server_process.wait()
        print(f"[INFO] Server completed with return code: {server_return_code}")

        # Wait for all clients to complete
        for i, client_process in enumerate(client_processes):
            client_return_code = client_process.wait()
            print(f"[INFO] Client {i} completed with return code: {client_return_code}")

        if server_return_code == 0:
            print("[INFO] Federated training completed successfully!")
        else:
            print(f"[WARNING] Server completed with non-zero return code: {server_return_code}")

    except Exception as e:
        print(f"[ERROR] Federated training failed: {e}")
        raise e

def centralized_test(test_dir, batch_size=32, num_scenarios=-1, visualize_limit=10, model_name="GATv2", seq_len=30):
    """Enhanced centralized testing with consolidated results"""
    test_loader = get_pyg_data_loader(test_dir, batch_size, num_scenarios, shuffle=False, mode='test', seq_len=seq_len)

    if not test_loader:
        print("[ERROR] Could not create test data loader.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, in_channels=5, hidden_channels=32, out_channels=2, seq_len=seq_len).to(device)
    
    model_path = f"saved_models/centralized_model_{model_name}.pt"
    if not os.path.exists(model_path):
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_loss, test_min_ade, test_min_fde, test_mr = evaluate(model, test_loader, device, seq_len=seq_len)
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "centralized_test_metrics.json")
    
    all_metrics = []
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            try:
                loaded_content = json.load(f)
                if isinstance(loaded_content, list):
                    all_metrics = loaded_content
                elif isinstance(loaded_content, dict):
                    all_metrics = [loaded_content]
                else:
                    all_metrics = []
            except json.JSONDecodeError:
                all_metrics = []

    run_metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "test_loss": test_loss,
        "test_min_ade_k1": test_min_ade,
        "test_min_fde_k1": test_min_fde,
        "test_mr_2m": test_mr
    }
    all_metrics.append(run_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    save_dir = os.path.join("results", "test_predictions", "centralized", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Generate visualizations and collect prediction data
    all_predictions = []
    vis_loader = get_pyg_data_loader(test_dir, batch_size=1, num_scenarios=visualize_limit, shuffle=False, mode='test', seq_len=seq_len)

    print(f"[INFO] Generating {visualize_limit} visualizations for centralized {model_name}...")
    for batch_idx, batch in enumerate(vis_loader):
        prediction_data = visualize_predictions(
            batch=batch,
            model=model,
            device=device,
            save_dir=save_dir,
            prefix=f"test_sample_{batch_idx}_",
            is_test_mode=True,
            seq_len=seq_len
        )
        if prediction_data:
            all_predictions.append(prediction_data)

    # Create consolidated predictions file
    consolidated_data = {
        'model_name': model_name,
        'training_type': 'centralized',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_metrics': {
            'test_loss': test_loss,
            'test_min_ade_k1': test_min_ade,
            'test_min_fde_k1': test_min_fde,
            'test_mr_2m': test_mr
        },
        'total_samples': len(all_predictions),
        'predictions': all_predictions,
        'summary_stats': {
            'avg_prediction_confidence': float(np.mean([p.get('model_confidence', 0) for p in all_predictions])) if all_predictions else 0.0,
            'samples_with_gt': len([p for p in all_predictions if p.get('ground_truth') is not None]),
            'avg_ade': float(np.mean([p['metrics']['ade'] for p in all_predictions if p.get('metrics') and p['metrics'].get('ade') is not None])) if any(p.get('metrics') and p['metrics'].get('ade') is not None for p in all_predictions) else 0.0,
            'avg_fde': float(np.mean([p['metrics']['fde'] for p in all_predictions if p.get('metrics') and p['metrics'].get('fde') is not None])) if any(p.get('metrics') and p['metrics'].get('fde') is not None for p in all_predictions) else 0.0
        }
    }

    consolidated_path = os.path.join(save_dir, 'consolidated_predictions.json')
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    print(f"[INFO] Saved consolidated predictions to: {consolidated_path}")

    # Clean up individual prediction files
    import glob
    individual_files = glob.glob(os.path.join(save_dir, 'test_sample_*_predictions.json'))
    for file in individual_files:
        try:
            os.remove(file)
        except:
            pass
    print(f"[INFO] Cleaned up {len(individual_files)} individual prediction files")

def federated_test(test_dir, batch_size=32, num_scenarios=-1, visualize_limit=5, model_name="GATv2", seq_len=30):
    test_loader = get_pyg_data_loader(test_dir, batch_size, num_scenarios, shuffle=False, mode='test', seq_len=seq_len)
    
    if not test_loader:
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name, in_channels=5, hidden_channels=32, out_channels=2, seq_len=seq_len).to(device)
    model_path = f"saved_models/federated_model_{model_name}.pt"
    if not os.path.exists(model_path):
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_loss, test_min_ade, test_min_fde, test_mr = evaluate(model, test_loader, device, seq_len=seq_len)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "federated_test_metrics.json")
    
    all_metrics = []
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            try:
                loaded_content = json.load(f)
                if isinstance(loaded_content, list):
                    all_metrics = loaded_content
                elif isinstance(loaded_content, dict):
                    all_metrics = [loaded_content]
                else:
                    all_metrics = []
            except json.JSONDecodeError:
                all_metrics = []

    run_metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "test_loss": test_loss,
        "test_min_ade_k1": test_min_ade,
        "test_min_fde_k1": test_min_fde,
        "test_mr_2m": test_mr
    }
    all_metrics.append(run_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    save_dir = os.path.join("results", "test_predictions", "federated", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Generate visualizations and collect prediction data
    all_predictions = []
    vis_loader = get_pyg_data_loader(test_dir, batch_size=1, num_scenarios=visualize_limit, shuffle=False, mode='test', seq_len=seq_len)

    print(f"[INFO] Generating {visualize_limit} visualizations for federated {model_name}...")
    for batch_idx, batch in enumerate(vis_loader):
        prediction_data = visualize_predictions(
            batch=batch,
            model=model,
            device=device,
            save_dir=save_dir,
            prefix=f"test_sample_{batch_idx}_",
            is_test_mode=True,
            seq_len=seq_len
        )
        if prediction_data:
            all_predictions.append(prediction_data)

    # Create consolidated predictions file
    consolidated_data = {
        'model_name': model_name,
        'training_type': 'federated',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_metrics': {
            'test_loss': test_loss,
            'test_min_ade_k1': test_min_ade,
            'test_min_fde_k1': test_min_fde,
            'test_mr_2m': test_mr
        },
        'total_samples': len(all_predictions),
        'predictions': all_predictions,
        'summary_stats': {
            'avg_prediction_confidence': float(np.mean([p.get('model_confidence', 0) for p in all_predictions])) if all_predictions else 0.0,
            'samples_with_gt': len([p for p in all_predictions if p.get('ground_truth') is not None]),
            'avg_ade': float(np.mean([p['metrics']['ade'] for p in all_predictions if p.get('metrics') and p['metrics'].get('ade') is not None])) if any(p.get('metrics') and p['metrics'].get('ade') is not None for p in all_predictions) else 0.0,
            'avg_fde': float(np.mean([p['metrics']['fde'] for p in all_predictions if p.get('metrics') and p['metrics'].get('fde') is not None])) if any(p.get('metrics') and p['metrics'].get('fde') is not None for p in all_predictions) else 0.0
        }
    }

    consolidated_path = os.path.join(save_dir, 'consolidated_predictions.json')
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    print(f"[INFO] Saved consolidated federated predictions to: {consolidated_path}")

    # Clean up individual prediction files
    import glob
    individual_files = glob.glob(os.path.join(save_dir, 'test_sample_*_predictions.json'))
    for file in individual_files:
        try:
            os.remove(file)
        except:
            pass
    print(f"[INFO] Cleaned up {len(individual_files)} individual prediction files")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['centralized', 'federated', 'test'], required=True)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_scenarios', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--role', choices=['server', 'client'])
    parser.add_argument('--model_name', type=str, default="GATv2")
    args = parser.parse_args()

    if args.mode == 'centralized':
        centralized_train(args.train_dir, args.val_dir, args.batch_size, args.num_scenarios, args.epochs, args.lr, model_name=args.model_name)
    elif args.mode == 'test':
        if not args.test_dir:
            print('Please specify --test_dir for test mode.')
            return
        centralized_test(args.test_dir, args.batch_size, args.num_scenarios, visualize_limit=5, model_name=args.model_name)
    elif args.mode == 'federated':
        if args.role == 'server':
            from federated.server import main as server_main
            server_main(model_name=args.model_name)
        elif args.role == 'client':
            from federated.client import main as client_main
            client_main(model_name=args.model_name)

if __name__ == "__main__":
    main()