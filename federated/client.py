import flwr as fl
import torch
import argparse
import os
import numpy as np
from models.gnn import GATv2TrajectoryPredictor
from models.vectornet import VectorNet
from utils.data_utils import get_pyg_data_loader
from collections import OrderedDict

def get_model(model_name, in_channels, hidden_channels, out_channels, seq_len=30):
    if model_name == "GATv2":
        return GATv2TrajectoryPredictor(in_channels, hidden_channels, out_channels * seq_len, seq_len=seq_len)
    elif model_name == "VectorNet":
        return VectorNet(in_channels, out_channels * seq_len, hidden_dim=hidden_channels, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device, model_name):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.model_name = model_name
        self.model_saved = False  # Flag to track if model has been saved

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        learning_rate = config.get("learning_rate", 0.001)
        epochs = config.get("epochs", 1)
        loss, min_ade_k1, min_fde_k1, mr_2m = train(self.model, self.trainloader, self.valloader, epochs=epochs, lr=learning_rate, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": float(loss), "min_ade_k1": float(min_ade_k1), "min_fde_k1": float(min_fde_k1), "mr_2m": float(mr_2m)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if self.valloader is not None:
            loss, min_ade_k1, min_fde_k1, mr_2m = test(self.model, self.valloader, device=self.device)
            dataset_size = len(self.valloader.dataset)
        else:
            # No validation data available
            loss, min_ade_k1, min_fde_k1, mr_2m = 0.0, 0.0, 0.0, 0.0
            dataset_size = 0

        # Only save the model at the very end of federated training
        # Check if this is the final evaluation of the final round
        current_round = config.get("current_round", 1)
        total_rounds = config.get("total_rounds", 1)
        
        # Debug: Print round information (only once per round to avoid spam)
        if not hasattr(self, 'last_round_logged') or self.last_round_logged != current_round:
            print(f"[Client DEBUG] Round {current_round}/{total_rounds} - Model saved: {self.model_saved}")
            print(f"[Client DEBUG] Config received: {config}")
            self.last_round_logged = current_round
        
        # Only save once at the very end
        if current_round == total_rounds and not self.model_saved:
            model_path = f"saved_models/federated_model_{self.model_name}.pt"
            torch.save(self.model.state_dict(), model_path)
            print(f"[Client INFO] ✅ Federated model saved to: {model_path}")
            print(f"[Client INFO] Training completed after {total_rounds} rounds")
            self.model_saved = True  # Mark as saved to prevent duplicate saves
        return float(loss), dataset_size, {"loss": float(loss), "min_ade_k1": float(min_ade_k1), "min_fde_k1": float(min_fde_k1), "mr_2m": float(mr_2m)}

def train(model, trainloader, valloader, epochs, lr, device):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in trainloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            outputs = outputs.view(-1, model.seq_len, 2)
            labels = batch.y.view(-1, model.seq_len, 2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Only test if validation loader is available
    if valloader is not None:
        return test(model, valloader, device)
    else:
        return 0.0, 0.0, 0.0, 0.0

def test(model, testloader, device):
    criterion = torch.nn.MSELoss()
    model.eval()
    total_loss = 0
    all_displacement_errors = []
    with torch.no_grad():
        for batch in testloader:
            batch = batch.to(device)
            outputs = model(batch)
            outputs = outputs.view(-1, model.seq_len, 2)
            labels = batch.y.view(-1, model.seq_len, 2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred_np = outputs.cpu().numpy()
            gt_np = labels.cpu().numpy()
            displacement_error = np.linalg.norm(pred_np - gt_np, axis=1)
            all_displacement_errors.extend(displacement_error.tolist())
    avg_loss = total_loss / len(testloader)
    min_ade_k1 = np.mean(all_displacement_errors)
    min_fde_k1 = np.mean(all_displacement_errors)
    mr_2m = np.mean(np.array(all_displacement_errors) > 2.0)
    return avg_loss, min_ade_k1, min_fde_k1, mr_2m

def main(model_name="GATv2", seq_len=30):
    parser = argparse.ArgumentParser(description="Flower Federated Learning Client")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--num_scenarios", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="GATv2")
    parser.add_argument("--seq_len", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_name, in_channels=5, hidden_channels=32, out_channels=2, seq_len=args.seq_len).to(device)
    print(f"[Client DEBUG] Initialized model: {args.model_name}")
    model_path = f"saved_models/federated_model_{args.model_name}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[Client INFO] Using existing model to train further: {model_path}")
    else:
        print("[Client INFO] Model not found, starting fresh.")
    
    # Use robust data loading with fallback for federated training
    train_loader = get_pyg_data_loader(
        args.train_dir, batch_size=args.batch_size, num_scenarios=args.num_scenarios,
        shuffle=True, mode='train', client_id=args.client_id, num_clients=args.num_clients,
        seq_len=args.seq_len, use_multiprocessing=True
    )
    val_loader = get_pyg_data_loader(
        args.val_dir, batch_size=args.batch_size,
        num_scenarios=min(args.num_scenarios, 200) if args.num_scenarios > 0 else 200,
        shuffle=False, mode='val', seq_len=args.seq_len, use_multiprocessing=True
    )

    if not train_loader:
        print("[Client ERROR] Could not create training data loader. Aborting client.")
        return

    if not val_loader:
        print("[Client WARNING] Could not create validation data loader. Continuing without validation.")
        val_loader = None

    client = FlowerClient(model, train_loader, val_loader, device, args.model_name)

    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client(), transport="grpc-bidi")

if __name__ == "__main__":
    main()