import flwr as fl
import torch
import argparse
from models.gnn import GATv2TrajectoryPredictor
from models.vectornet import VectorNet # Import VectorNet
from collections import OrderedDict
import json
import os
from datetime import datetime # Import datetime

def get_model(model_name, in_channels, hidden_channels, out_channels):
    if model_name == "GATv2":
        return GATv2TrajectoryPredictor(in_channels, hidden_channels, out_channels)
    elif model_name == "VectorNet":
        return VectorNet(in_channels, out_channels, hidden_dim=hidden_channels)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def weighted_average(metrics):
    """Aggregate metrics manually."""
    # Multiply each metric by the number of examples
    weighted_metrics = [(num_examples, m) for num_examples, m in metrics]
    
    # Aggregate loss
    losses = [num_examples * m["loss"] for num_examples, m in weighted_metrics]
    total_examples = sum(num_examples for num_examples, _ in weighted_metrics)
    
    if total_examples == 0:
        return {"loss": float("inf")}  # Or handle as you see fit
        
    avg_loss = sum(losses) / total_examples
    
    # Aggregate new metrics
    min_ade_k1s = [num_examples * m["min_ade_k1"] for num_examples, m in weighted_metrics]
    min_fde_k1s = [num_examples * m["min_fde_k1"] for num_examples, m in weighted_metrics]
    mr_2ms = [num_examples * m["mr_2m"] for num_examples, m in weighted_metrics]

    avg_min_ade_k1 = sum(min_ade_k1s) / total_examples
    avg_min_fde_k1 = sum(min_fde_k1s) / total_examples
    avg_mr_2m = sum(mr_2ms) / total_examples
    
    return {"loss": avg_loss, "min_ade_k1": avg_min_ade_k1, "min_fde_k1": avg_min_fde_k1, "mr_2m": avg_mr_2m}

def get_on_fit_config_fn(config):
    """Return a function which returns training configurations."""
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        return {
            "learning_rate": config.lr,
            "current_round": server_round,
            "epochs": config.client_epochs
        }
    return fit_config

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for clients.")
    parser.add_argument("--min_clients", type=int, default=2, help="Minimum number of clients.")
    parser.add_argument("--client_epochs", type=int, default=1, help="Number of epochs for client training.")
    parser.add_argument("--model_name", type=str, default="GATv2", help="Model to use: GATv2 or VectorNet")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("results", exist_ok=True)

    print("[INFO] Starting Flower server...")
    print(f"[INFO] Minimum clients required: {args.min_clients}")
    print(f"[INFO] Number of rounds: {args.rounds}")
    
    # Define model for parameter initialization
    model = get_model(args.model_name, in_channels=5, hidden_channels=32, out_channels=2)
    model_path = f"saved_models/federated_model_{args.model_name}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[INFO] Loaded existing model from {model_path}")
    
    # Define strategy with more lenient client requirements
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=get_on_fit_config_fn(args),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()]),
    )

    # Start server with longer timeouts
    try:
        server_config = fl.server.ServerConfig(
            num_rounds=args.rounds,
            round_timeout=600.0  # 10 minutes round timeout
        )
        
        history = fl.server.start_server(
            server_address="127.0.0.1:8080",  # IPv6 address that also accepts IPv4
            config=server_config,
            strategy=strategy,
        )

        print("\n[INFO] Federated training complete.")
        
        # Save federated training history if available
        if hasattr(history, 'losses_distributed') and history.losses_distributed:
            # Extract final aggregated metrics from the last round
            final_metrics = {}
            if history.metrics_distributed:
                for metric_name, metric_values_list in history.metrics_distributed.items():
                    if metric_values_list: # Ensure the list is not empty
                        final_metrics[metric_name] = metric_values_list[-1][1] # Get the value from the last tuple

            history_path = os.path.join("results", "federated_training_history.json")
            
            # Load existing history or initialize an empty list
            all_histories = []
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    try:
                        all_histories = json.load(f)
                    except json.JSONDecodeError: # Handle empty or malformed JSON
                        all_histories = []

            # Append current run's history with metadata and final metrics
            run_metadata = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": args.model_name, # Use args.model_name here
                "rounds_trained": args.rounds,
                "client_epochs": args.client_epochs,
                "final_loss": history.losses_distributed[-1][1] if history.losses_distributed else None,
                "final_metrics": final_metrics
            }
            all_histories.append(run_metadata)

            with open(history_path, 'w') as f:
                json.dump(all_histories, f, indent=4)
            print(f"[INFO] Federated training history saved to {history_path}")
            print(f"[INFO] Final aggregated loss: {history.losses_distributed[-1][1] if history.losses_distributed else 'N/A'}")
        
        # Save the final model if parameters are available
        # (Removed: strategy.parameters is not available in recent Flower versions)
        # Saving the final model should be done by a client after the last round.
            
    except Exception as e:
        print(f"[ERROR] Server failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()