import hashlib
import json
import time

class BlockchainLedger:
    def __init__(self, ledger_file="ledger.json"):
        self.ledger_file = ledger_file
        self.chain = self._load_ledger()

    def _load_ledger(self):
        try:
            with open(self.ledger_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_ledger(self):
        with open(self.ledger_file, 'w') as f:
            json.dump(self.chain, f, indent=4)

    def calculate_hash(self, data):
        """
        Generates SHA-256 hash of the canonical JSON representation.
        Ensures dictionary keys are sorted for consistent hashing.
        """
        canonical_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def add_record(self, structured_data, metadata=None):
        """
        Adds a record to the tamper-evident ledger.
        Stores ONLY the hash, timestamp, and minimal metadata.
        """
        content_hash = self.calculate_hash(structured_data)
        
        # Check if already exists
        for block in self.chain:
            if block['hash'] == content_hash:
                return block # Return existing block

        # Create new block
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time.time(),
            "timestamp_human": time.ctime(),
            "hash": content_hash,
            "metadata": metadata or {},
            "previous_hash": self.chain[-1]['hash'] if self.chain else "0"
        }
        
        self.chain.append(block)
        self._save_ledger()
        return block

    def verify_record(self, structured_data):
        """
        Verifies if the data exists in the ledger and hasn't been tampered with.
        """
        current_hash = self.calculate_hash(structured_data)
        for block in self.chain:
            if block['hash'] == current_hash:
                return True, block
        return False, None
