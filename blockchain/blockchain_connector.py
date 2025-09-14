import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from web3 import Web3


class BlockchainLogger:
    """Web3.py helper to interact with ModelUpdates smart contract."""
    def __init__(self, rpc_url: str, contract_address: str, abi_path: Path, private_key: str | None = None):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        self.addr = Web3.to_checksum_address(contract_address)
        with open(abi_path, "r", encoding="utf-8") as f:
            abi = json.load(f)
        self.contract = self.web3.eth.contract(address=self.addr, abi=abi)
        self.private_key = private_key
        self.account = None
        if private_key:
            self.account = self.web3.eth.account.from_key(private_key)

    def record_update(self, update_hash_hex: str, accuracy_bp: int, hospital_id: str) -> str:
        """Send transaction to record model update.
        Returns tx hash hex string.
        """
        if not self.account:
            raise RuntimeError("Private key not set for sending transactions")
        nonce = self.web3.eth.get_transaction_count(self.account.address)
        tx = self.contract.functions.recordUpdate(Web3.to_bytes(hexstr=update_hash_hex), int(accuracy_bp), hospital_id).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 500000,
            "maxFeePerGas": self.web3.to_wei("2", "gwei"),
            "maxPriorityFeePerGas": self.web3.to_wei("1", "gwei"),
        })
        signed = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()

    def get_update_history(self) -> List[Tuple[str, int, str, int]]:
        """Return list of (hash_hex, accuracy_bp, hospital_id, timestamp)."""
        count = self.contract.functions.getUpdateCount().call()
        out = []
        for i in range(count):
            h, acc_bp, hosp, ts = self.contract.functions.getUpdate(i).call()
            out.append((h.hex(), int(acc_bp), str(hosp), int(ts)))
        return out


