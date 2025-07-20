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
        loss, min_ade_k1, min_fde_k1, mr_2m = test(self.model, self.valloader, device=self.device)
        if config.get("current_round", 1) == config.get("total_rounds", 1):
            model_path = f"saved_models/federated_model_{self.model_name}.pt"
            torch.save(self.model.state_dict(), model_path)
        return float(loss), len(self.valloader.dataset), {"loss": float(loss), "min_ade_k1": float(min_ade_k1), "min_fde_k1": float(min_fde_k1), "mr_2m": float(mr_2m)}

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
    return test(model, valloader, device)

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
    
    train_loader = get_pyg_data_loader(args.train_dir, batch_size=args.batch_size, num_scenarios=args.num_scenarios, shuffle=True, mode='train', client_id=args.client_id, num_clients=args.num_clients, seq_len=args.seq_len)
    val_loader = get_pyg_data_loader(args.val_dir, batch_size=args.batch_size, num_scenarios=args.num_scenarios, shuffle=False, mode='val', seq_len=args.seq_len)

    client = FlowerClient(model, train_loader, val_loader, device, args.model_name)
    
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client(), transport="grpc-bidi")

if __name__ == "__main__":
    main()