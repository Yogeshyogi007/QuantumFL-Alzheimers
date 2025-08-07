import argparse
import json
import time
from pathlib import Path
from web3 import Web3

GANACHE_URL = "http://127.0.0.1:7545"  # Default Ganache
CONTRACT_ABI_PATH = Path(__file__).parent / 'contract_abi.json'
CONTRACT_ADDRESS_PATH = Path(__file__).parent / 'contract_address.txt'

def load_contract(w3):
    with open(CONTRACT_ABI_PATH) as f:
        abi = json.load(f)
    with open(CONTRACT_ADDRESS_PATH) as f:
        address = f.read().strip()
    contract = w3.eth.contract(address=address, abi=abi)
    return contract

def log_update(w3, contract, account, model_hash, accuracy):
    tx_hash = contract.functions.logUpdate(model_hash, float(accuracy), int(time.time())).transact({'from': account})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Logged update:', receipt)

def query_history(contract):
    events = contract.events.UpdateLogged.createFilter(fromBlock=0).get_all_entries()
    for e in events:
        print(f"Model Hash: {e['args']['modelHash']}, Accuracy: {e['args']['accuracy']}, Timestamp: {e['args']['timestamp']}")

def main():
    parser = argparse.ArgumentParser(description='Blockchain logger for FL model updates.')
    parser.add_argument('--log', action='store_true', help='Log a new model update')
    parser.add_argument('--hash', type=str, help='Model update hash')
    parser.add_argument('--acc', type=float, help='Model accuracy')
    parser.add_argument('--history', action='store_true', help='Query update history')
    args = parser.parse_args()
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    if not w3.isConnected():
        print('Could not connect to Ganache.')
        return
    contract = load_contract(w3)
    account = w3.eth.accounts[0]
    if args.log:
        if not args.hash or args.acc is None:
            print('Provide --hash and --acc to log update.')
            return
        log_update(w3, contract, account, args.hash, args.acc)
    if args.history:
        query_history(contract)

if __name__ == "__main__":
    main()