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
    train_loader = get_pyg_data_loader(train_dir, batch_size, num_scenarios, shuffle=True, mode='train', seq_len=seq_len)
    val_loader = get_pyg_data_loader(val_dir, batch_size, num_scenarios, shuffle=False, mode='val', seq_len=seq_len)

    if not train_loader or not val_loader:
        print("[ERROR] Could not create data loaders. Aborting training.")
        return

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
        val_loss, val_min_ade, val_min_fde, val_mr = evaluate(model, val_loader, device, seq_len=seq_len)
        
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

def centralized_test(test_dir, batch_size=32, num_scenarios=-1, visualize_limit=5, model_name="GATv2", seq_len=30):
    test_loader = get_pyg_data_loader(test_dir, batch_size, num_scenarios, shuffle=False, mode='test', seq_len=seq_len)
    
    if not test_loader:
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

    vis_loader = get_pyg_data_loader(test_dir, batch_size=1, num_scenarios=visualize_limit, shuffle=False, mode='test', seq_len=seq_len)
    for batch_idx, batch in enumerate(vis_loader):
        visualize_predictions(
            batch=batch, 
            model=model, 
            device=device, 
            save_dir=save_dir,
            prefix=f"test_sample_{batch_idx}_",
            is_test_mode=True,
            seq_len=seq_len
        )

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

    vis_loader = get_pyg_data_loader(test_dir, batch_size=1, num_scenarios=visualize_limit, shuffle=False, mode='test', seq_len=seq_len)
    for batch_idx, batch in enumerate(vis_loader):
        visualize_predictions(
            batch=batch, 
            model=model, 
            device=device, 
            save_dir=save_dir,
            prefix=f"test_sample_{batch_idx}_",
            is_test_mode=True,
            seq_len=seq_len
        )

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