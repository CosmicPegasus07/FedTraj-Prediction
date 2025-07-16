import argparse
import os
import torch
import json # Import json module
from models.gnn import GATTrajectoryPredictor
from utils.data_utils import get_pyg_data_loader
from utils.viz_utils import visualize_predictions # Import the function

def evaluate(model, loader, device):
    """
    Evaluates the model on the provided data loader.
    """
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)
            
            # Get focal agent predictions
            focal_global_idx = batch.ptr[:-1] + batch.focal_idx
            focal_outputs = outputs[focal_global_idx] # This will be [batch_size, 2]
            
            labels = batch.y
            # Reshape labels to match focal_outputs [batch_size, 2]
            labels = labels.view(-1, 2) # This will transform [4] to [2, 2] for batch_size=2
            
            loss = criterion(focal_outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[INFO] Validation/Test Loss: {avg_loss:.4f}")
    return avg_loss

def centralized_train(train_dir, val_dir, batch_size=32, num_scenarios=-1, epochs=5, lr=1e-3):
    """
    Performs centralized training.
    """
    print("[INFO] Starting centralized training...")
    print(f"[INFO] Using train set from: {train_dir}")
    print(f"[INFO] Using validation set from: {val_dir}")
    train_loader = get_pyg_data_loader(train_dir, batch_size, num_scenarios, shuffle=True, mode='train')
    val_loader = get_pyg_data_loader(val_dir, batch_size, num_scenarios, shuffle=False, mode='val')

    if not train_loader or not val_loader:
        print("[ERROR] Could not create data loaders. Aborting training.")
        return

    print(f"[INFO] Loaded {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} val samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATTrajectoryPredictor(in_channels=5, hidden_channels=32, out_channels=2).to(device)

    model_path = "centralized_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[INFO] Loaded existing model from {model_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    history = [] # Initialize history list
    for epoch in range(epochs):
        print(f"[INFO] Epoch {epoch+1}/{epochs} starting...")
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch.x, batch.edge_index)

            # Get focal agent predictions (this part is now correct for any batch_size > 0)
            focal_global_idx = batch.ptr[:-1] + batch.focal_idx
            focal_outputs = outputs[focal_global_idx] # This will be [batch_size, 2]

            labels = batch.y
            # Reshape labels to match focal_outputs [batch_size, 2]
            # Use .view(-1, 2) to reshape it to (number_of_samples_in_batch, 2)
            labels = labels.view(-1, 2) # This will transform [4] to [2, 2] for batch_size=2

            loss = criterion(focal_outputs, labels)
            # ... rest of the training loop ...
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                 print(f"    [Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = total_loss/len(train_loader)
        print(f"[INFO] Epoch {epoch+1} complete. Avg Train Loss: {avg_train_loss:.4f}")
        print("[INFO] Running validation...")
        val_loss = evaluate(model, val_loader, device)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        })

    print("[INFO] Centralized training complete.")
    torch.save(model.state_dict(), "centralized_model.pt")
    print("[INFO] Model saved as centralized_model.pt")

    # Save training history to JSON
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "centralized_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"[INFO] Training history saved to {history_path}")


# ... (evaluate function, centralized_train function) ...

def centralized_test(test_dir, batch_size=32, num_scenarios=-1, visualize_limit=5):
    """
    Performs centralized testing and visualization.
    """
    print("[INFO] Starting centralized testing...")
    print(f"[INFO] Using test set from: {test_dir}")
    # Ensure mode='test' is passed to get_pyg_data_loader
    test_loader = get_pyg_data_loader(test_dir, batch_size, num_scenarios, shuffle=False, mode='test')
    
    if not test_loader:
        print("[ERROR] Could not create test data loader. Aborting test.")
        return

    print(f"[INFO] Loaded {len(test_loader.dataset)} test samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATTrajectoryPredictor(in_channels=5, hidden_channels=32, out_channels=2).to(device)
    
    model_path = "centralized_model.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Loaded model weights from {model_path}")
    
    # Run evaluation to get the test loss (if you want to report it)
    test_loss = evaluate(model, test_loader, device) # Evaluate will now get placeholder y for test
    print(f"[INFO] Final test loss: {test_loss:.4f}")
    
    # --- Add Visualization ---
    print("[INFO] Generating visualizations for test predictions...")
    save_dir = os.path.join("results", "test_predictions", "centralized")
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through a few batches (or all) for visualization
    for batch_idx, batch in enumerate(test_loader):
        # You might want to limit the number of visualizations if your test set is huge
        if batch_idx >= visualize_limit: # Visualize only the first N batches, adjust as needed
            break 
        visualize_predictions(
            batch=batch, 
            model=model, 
            device=device, 
            save_dir=save_dir, # Create a specific directory for test results
            prefix=f"test_batch_{batch_idx}_",
            is_test_mode=True # Indicate that it's test mode to skip ground truth target
        )
    print(f"[INFO] Visualizations complete. Check '{save_dir}' folder.")
    
    print("[INFO] Centralized testing complete.")

def federated_test(test_dir, batch_size=32, num_scenarios=-1, visualize_limit=5):
    """
    Performs federated testing and visualization.
    """
    print("[INFO] Starting federated testing...")
    print(f"[INFO] Using test set from: {test_dir}")
    test_loader = get_pyg_data_loader(test_dir, batch_size, num_scenarios, shuffle=False, mode='test')
    
    if not test_loader:
        print("[ERROR] Could not create test data loader. Aborting test.")
        return

    print(f"[INFO] Loaded {len(test_loader.dataset)} test samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATTrajectoryPredictor(in_channels=5, hidden_channels=32, out_channels=2).to(device)
    
    model_path = "federated_model.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}. Please run federated training first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Loaded model weights from {model_path}")
    
    test_loss = evaluate(model, test_loader, device)
    print(f"[INFO] Final federated test loss: {test_loss:.4f}")

    # Save federated test metrics to JSON
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "federated_test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({'test_loss': test_loss}, f, indent=4)
    print(f"[INFO] Federated test metrics saved to {metrics_path}")
    
    print("[INFO] Generating visualizations for federated test predictions...")
    save_dir = "results/test_predictions/federated"
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= visualize_limit:
            break 
        visualize_predictions(
            batch=batch, 
            model=model, 
            device=device, 
            save_dir=save_dir,
            prefix=f"test_batch_{batch_idx}_",
            is_test_mode=True
        )
    print(f"[INFO] Visualizations complete. Check '{save_dir}' folder.")
    
    print("[INFO] Federated testing complete.")

# ... (main function) ...

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['centralized', 'federated', 'test'], required=True, help='Training mode: centralized, federated, or test')
    parser.add_argument('--train_dir', type=str, help='Directory with training scenario folders')
    parser.add_argument('--val_dir', type=str, help='Directory with validation scenario folders')
    parser.add_argument('--test_dir', type=str, help='Directory with test scenario folders (for test mode)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_scenarios', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--role', choices=['server', 'client'], help='Role to run: server or client (for federated only)')
    args = parser.parse_args()

    if args.mode == 'centralized':
        if not args.train_dir or not args.val_dir:
            print('Please specify both --train_dir and --val_dir for centralized training.')
            return
        centralized_train(args.train_dir, args.val_dir, args.batch_size, args.num_scenarios, args.epochs, args.lr)
    elif args.mode == 'test':
        if not args.test_dir:
            print('Please specify --test_dir for test mode.')
            return
        # Pass visualize_limit if available, else default to 5
        centralized_test(args.test_dir, args.batch_size, args.num_scenarios, visualize_limit=5)
    elif args.mode == 'federated':
        if args.role == 'server':
            from federated.server import main as server_main
            server_main()
        elif args.role == 'client':
            from federated.client import main as client_main
            client_main()
        else:
            print('Please specify --role as server or client for federated mode.')

if __name__ == "__main__":
    main()
