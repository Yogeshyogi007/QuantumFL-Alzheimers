import flwr as fl
import argparse
from pathlib import Path
from typing import Dict, Optional


def weighted_average(metrics):
    # Example strategy metric aggregation (accuracy)
    accuracies = [m[1]["accuracy"] for m in metrics if "accuracy" in m[1]]
    if not accuracies:
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / len(accuracies)}

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Server (Flower)')
    parser.add_argument('--rounds', type=int, default=3, help='Number of FL rounds')
    args = parser.parse_args()
    # Start Flower server with FedAvg
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=args.rounds), strategy=strategy)

if __name__ == "__main__":
    main()