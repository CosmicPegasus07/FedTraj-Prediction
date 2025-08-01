import flwr as fl
import torch
import argparse
from models.gnn import GATv2TrajectoryPredictor
from models.vectornet import VectorNet
from collections import OrderedDict
import json
import os
from datetime import datetime
import tempfile

def get_model(model_name, in_channels, hidden_channels, out_channels, seq_len=30):
    if model_name == "GATv2":
        return GATv2TrajectoryPredictor(in_channels, hidden_channels, out_channels * seq_len, seq_len=seq_len)
    elif model_name == "VectorNet":
        return VectorNet(in_channels, out_channels * seq_len, hidden_dim=hidden_channels, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def weighted_average(metrics):
    weighted_metrics = [(num_examples, m) for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in weighted_metrics)
    if total_examples == 0:
        return {"loss": float("inf")}
    avg_loss = sum(num_examples * m["loss"] for num_examples, m in weighted_metrics) / total_examples
    avg_min_ade_k1 = sum(num_examples * m["min_ade_k1"] for num_examples, m in weighted_metrics) / total_examples
    avg_min_fde_k1 = sum(num_examples * m["min_fde_k1"] for num_examples, m in weighted_metrics) / total_examples
    avg_mr_2m = sum(num_examples * m["mr_2m"] for num_examples, m in weighted_metrics) / total_examples
    return {"loss": avg_loss, "min_ade_k1": avg_min_ade_k1, "min_fde_k1": avg_min_fde_k1, "mr_2m": avg_mr_2m}

# Global variable to store per-round metrics for logging
round_training_history = []

def get_on_fit_config_fn(args):
    def fit_config(server_round: int):
        return {
            "learning_rate": args.lr,
            "epochs": args.client_epochs,
            "current_round": server_round,
            "total_rounds": args.rounds,
        }
    return fit_config

def get_on_evaluate_config_fn(args):
    def evaluate_config(server_round: int):
        return {
            "current_round": server_round,
            "total_rounds": args.rounds,
        }
    return evaluate_config

def fit_metrics_aggregation_fn(fit_metrics):
    """Custom fit metrics aggregation function that logs per-round training metrics"""
    global round_training_history
    
    # Calculate weighted average training metrics
    aggregated_metrics = weighted_average(fit_metrics)
    
    # Log the round training metrics
    round_data = {
        "round": len(round_training_history) + 1,
        "loss": aggregated_metrics["loss"],  # Frontend expects 'loss', not 'training_loss'
        "training_loss": aggregated_metrics["loss"],  # Keep both for compatibility
        "min_ade_k1": aggregated_metrics["min_ade_k1"],  # Frontend expects this format
        "min_fde_k1": aggregated_metrics["min_fde_k1"],  # Frontend expects this format
        "mr_2m": aggregated_metrics["mr_2m"],  # Frontend expects this format
        "training_min_ade_k1": aggregated_metrics["min_ade_k1"],  # Keep both for compatibility
        "training_min_fde_k1": aggregated_metrics["min_fde_k1"],  # Keep both for compatibility
        "training_mr_2m": aggregated_metrics["mr_2m"],  # Keep both for compatibility
        "num_clients": len(fit_metrics)
    }
    
    round_training_history.append(round_data)
    
    print(f"[Server INFO] Round {round_data['round']} Training - Loss: {round_data['loss']:.6f}, "
          f"ADE: {round_data['min_ade_k1']:.6f}, FDE: {round_data['min_fde_k1']:.6f}, "
          f"MR: {round_data['mr_2m']:.6f}")
    
    return aggregated_metrics

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for clients.")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients.")
    parser.add_argument("--client_epochs", type=int, default=1, help="Number of epochs for client training.")
    parser.add_argument("--model_name", type=str, default="GATv2", help="Model to use: GATv2 or VectorNet")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for trajectory prediction.")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    model = get_model(args.model_name, in_channels=5, hidden_channels=32, out_channels=2, seq_len=args.seq_len)
    print(f"[Server DEBUG] Initialized model: {args.model_name}")
    model_path = f"saved_models/federated_model_{args.model_name}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[Server INFO] Using existing model to train further: {model_path}")
    else:
        print("[Server INFO] Model not found, starting fresh.")

    # Track per-round metrics
    round_metrics = []

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=get_on_fit_config_fn(args),
        on_evaluate_config_fn=get_on_evaluate_config_fn(args),
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # Use custom function for logging
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()]),
    )

    server_config = fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=600.0)
    
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=server_config,
        strategy=strategy,
    )

    if hasattr(history, 'losses_distributed') and history.losses_distributed:
        final_metrics = {}
        if history.metrics_distributed:
            for metric_name, metric_values_list in history.metrics_distributed.items():
                if metric_values_list:
                    final_metrics[metric_name] = metric_values_list[-1][1]

        history_path = os.path.join("results", "federated_training_history.json")
        
        all_histories = []
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                try:
                    all_histories = json.load(f)
                except json.JSONDecodeError:
                    all_histories = []

        run_metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": args.model_name,
            "rounds_trained": args.rounds,
            "client_epochs": args.client_epochs,
            "final_loss": history.losses_distributed[-1][1] if history.losses_distributed else None,
            "final_metrics": final_metrics,
            "round_metrics": round_training_history,  # Frontend expects this field name
            "training_history": round_training_history  # Keep both for compatibility
        }
        all_histories.append(run_metadata)

        with open(history_path, 'w') as f:
            json.dump(all_histories, f, indent=4)

    # Create completion marker file to signal training is done
    completion_file = os.path.join("temp", "federated_training_completed.txt")
    os.makedirs("temp", exist_ok=True)
    with open(completion_file, 'w') as f:
        f.write("completed")
    print(f"[Server INFO] ✅ Federated training completed. Marker file created: {completion_file}")

if __name__ == "__main__":
    main()