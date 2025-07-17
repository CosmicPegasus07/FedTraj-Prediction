import flwr as fl
import torch
import argparse
import os
import numpy as np # Import numpy for metric calculations
from models.gnn import GATv2TrajectoryPredictor
from models.vectornet import VectorNet # Import VectorNet
from utils.data_utils import get_pyg_data_loader
from collections import OrderedDict
import time

def get_model(model_name, in_channels, hidden_channels, out_channels):
    if model_name == "GATv2":
        return GATv2TrajectoryPredictor(in_channels, hidden_channels, out_channels)
    elif model_name == "VectorNet":
        return VectorNet(in_channels, out_channels, hidden_dim=hidden_channels)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device, model_name):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.model_name = model_name

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"[Client ERROR] Failed to set parameters: {e}")
            raise

    def fit(self, parameters, config):
        try:
            print("[Client] Starting fit round")
            self.set_parameters(parameters)
            
            # Get current round number, learning rate, and epochs from config
            current_round = config.get("current_round", 1)
            learning_rate = config.get("learning_rate", 0.001)
            epochs = config.get("epochs", 1)
            print(f"[Client] Round {current_round}, LR: {learning_rate}, Epochs: {epochs}")
            
            # Train for the specified number of epochs
            loss, min_ade_k1, min_fde_k1, mr_2m = train(self.model, self.trainloader, self.valloader, epochs=epochs, lr=learning_rate, device=self.device)
            
            print(f"[Client] Fit round completed. Loss: {loss:.4f}, minADE(K=1): {min_ade_k1:.4f}, minFDE(K=1): {min_fde_k1:.4f}, MR(2.0m): {mr_2m:.4f}")
            return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": float(loss), "min_ade_k1": float(min_ade_k1), "min_fde_k1": float(min_fde_k1), "mr_2m": float(mr_2m)}
            
        except Exception as e:
            print(f"[Client ERROR] Fit failed: {e}")
            raise

    def evaluate(self, parameters, config):
        try:
            print("[Client] Starting evaluate round")
            self.set_parameters(parameters)
            loss, min_ade_k1, min_fde_k1, mr_2m = test(self.model, self.valloader, device=self.device)
            print(f"[Client] Evaluate completed. Loss: {loss:.4f}, minADE(K=1): {min_ade_k1:.4f}, minFDE(K=1): {min_fde_k1:.4f}, MR(2.0m): {mr_2m:.4f}")
            # Save the model after the final round if indicated by config
            if config.get("current_round", 1) == config.get("total_rounds", 1):
                # Use the model_name passed to the client for saving
                model_path = f"saved_models/federated_model_{self.model_name}.pt"
                print(f"[Client] Final round detected. Saving federated model as '{model_path}'.")
                print(f"[Client DEBUG] Model being saved: {type(self.model)}")
                torch.save(self.model.state_dict(), model_path)
            return float(loss), len(self.valloader.dataset), {"loss": float(loss), "min_ade_k1": float(min_ade_k1), "min_fde_k1": float(min_fde_k1), "mr_2m": float(mr_2m)}
        except Exception as e:
            print(f"[Client ERROR] Evaluate failed: {e}")
            raise

def train(model, trainloader, valloader, epochs, lr, device):
    """Train the model on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    batches = 0
    all_displacement_errors = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(trainloader):
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch.x, batch.edge_index)
                focal_global_idx = batch.ptr[:-1] + batch.focal_idx
                focal_outputs = outputs[focal_global_idx]
                labels = batch.y.view(-1, 2)
                loss = criterion(focal_outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1

                # Calculate displacement error for metrics during training as well
                pred_np = focal_outputs.detach().cpu().numpy()
                gt_np = labels.detach().cpu().numpy()
                displacement_error = np.linalg.norm(pred_np - gt_np, axis=1)
                all_displacement_errors.extend(displacement_error.tolist())
                    
            except Exception as e:
                print(f"[Client ERROR] Training batch {batch_idx} failed: {e}")
                continue
        
        val_loss, _, _, _ = test(model, valloader, device) # Only need loss for printing here
        print(f"[Client] Epoch {epoch + 1}/{epochs} completed. Average loss: {epoch_loss / batches if batches > 0 else float('inf'):.4f}, Val Loss: {val_loss:.4f}")
        total_loss += epoch_loss / batches if batches > 0 else float('inf')
        
    avg_loss = total_loss / epochs if epochs > 0 else float('inf')

    # Calculate additional metrics for the entire training run
    if all_displacement_errors:
        min_ade_k1 = np.mean(all_displacement_errors)
        min_fde_k1 = np.mean(all_displacement_errors) # For single point, ADE and FDE are the same
        mr_2m = np.mean(np.array(all_displacement_errors) > 2.0)
    else:
        min_ade_k1 = float('inf')
        min_fde_k1 = float('inf')
        mr_2m = float('inf')

    return avg_loss, min_ade_k1, min_fde_k1, mr_2m

def test(model, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.MSELoss()
    model.eval()
    total_loss = 0
    batches = 0
    all_displacement_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            try:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index)
                focal_global_idx = batch.ptr[:-1] + batch.focal_idx
                focal_outputs = outputs[focal_global_idx]
                labels = batch.y.view(-1, 2)
                loss = criterion(focal_outputs, labels)
                total_loss += loss.item()
                batches += 1

                # Calculate displacement error for metrics
                pred_np = focal_outputs.detach().cpu().numpy()
                gt_np = labels.detach().cpu().numpy()
                displacement_error = np.linalg.norm(pred_np - gt_np, axis=1) # Euclidean distance
                all_displacement_errors.extend(displacement_error.tolist())
                
            except Exception as e:
                print(f"[Client ERROR] Test batch {batch_idx} failed: {e}")
                continue
    
    avg_loss = total_loss / batches if batches > 0 else float('inf')

    # Calculate additional metrics
    if all_displacement_errors:
        min_ade_k1 = np.mean(all_displacement_errors)
        min_fde_k1 = np.mean(all_displacement_errors) # For single point, ADE and FDE are the same
        mr_2m = np.mean(np.array(all_displacement_errors) > 2.0)
    else:
        min_ade_k1 = float('inf')
        min_fde_k1 = float('inf')
        mr_2m = float('inf')

    return avg_loss, min_ade_k1, min_fde_k1, mr_2m

def main(model_name="GATv2"):
    parser = argparse.ArgumentParser(description="Flower Federated Learning Client")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with training scenario folders")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory with validation scenario folders")
    parser.add_argument("--num_scenarios", type=int, default=-1, help="Number of scenarios to load (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--num_clients", type=int, required=True, help="Total number of clients")
    parser.add_argument("--model_name", type=str, default="GATv2", help="Model to use: GATv2 or VectorNet")
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client] Using device: {device}")

    # Initialize model
    model = get_model(args.model_name, in_channels=5, hidden_channels=32, out_channels=2).to(device)
    print(f"[Client DEBUG] Model instantiated in client main: {type(model)}")
    
    # Load data
    print(f"[Client] Loading data from {args.train_dir} and {args.val_dir}")
    train_loader = get_pyg_data_loader(args.train_dir, batch_size=args.batch_size, 
                                     num_scenarios=args.num_scenarios, shuffle=True, mode='train', 
                                     client_id=args.client_id, num_clients=args.num_clients)
    val_loader = get_pyg_data_loader(args.val_dir, batch_size=args.batch_size, 
                                   num_scenarios=args.num_scenarios, shuffle=False, mode='val')

    if not train_loader or not val_loader:
        print("[Client ERROR] Failed to create data loaders")
        return

    print(f"[Client] Loaded {len(train_loader.dataset)} train samples and {len(val_loader.dataset)} val samples")

    # Create and start client
    client = FlowerClient(model, train_loader, val_loader, device, args.model_name)
    
    # Set longer timeouts for client
    fl.common.logger.configure(identifier="client")
    
    try:
        fl.client.start_client(
            server_address="127.0.0.1:8080",
            client=client.to_client(),
            transport="grpc-bidi"
        )
        print("[Client] Training completed successfully")
    except Exception as e:
        print(f"[Client ERROR] Failed to connect to server: {e}")
        return

if __name__ == "__main__":
    main()