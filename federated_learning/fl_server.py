import flwr as fl
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Server (Flower)')
    parser.add_argument('--rounds', type=int, default=3, help='Number of FL rounds')
    args = parser.parse_args()
    # Start Flower server
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=args.rounds))

if __name__ == "__main__":
    main()