import json
import hashlib
import os
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class PredictionRecord:
    drug_pair: tuple
    probability: float
    model_hash: str
    user_id: str
    timestamp: str
    risk_category: str
    explanation: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class BlockchainAuditLogger:
    def __init__(self, ledger_path: str = "./logs/prediction_ledger.json"):
        self.ledger_path = ledger_path
        self.ledger = self._load_ledger()
    
    def _load_ledger(self) -> list:
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_ledger(self):
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, 'w') as f:
            json.dump(self.ledger, f, indent=2)
    
    def _compute_hash(self, record: PredictionRecord) -> str:
        record_str = json.dumps(record.to_dict(), sort_keys=True)
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def log_prediction(self, drug_pair: tuple, probability: float, model_hash: str, user_id: str, risk_category: str, explanation: str) -> str:
        record = PredictionRecord(drug_pair=drug_pair, probability=probability, model_hash=model_hash, user_id=user_id, timestamp=datetime.utcnow().isoformat(), risk_category=risk_category, explanation=explanation)
        record_hash = self._compute_hash(record)
        record_dict = record.to_dict()
        record_dict['record_hash'] = record_hash
        self.ledger.append(record_dict)
        self._save_ledger()
        return record_hash
    
    def get_user_history(self, user_id: str, limit: int = 10) -> list:
        user_records = [r for r in self.ledger if r['user_id'] == user_id]
        return user_records[-limit:]
