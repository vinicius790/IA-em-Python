import json
from pathlib import Path
from datetime import datetime

class PredictionLogExporter:
    def __init__(self, output_dir: str = "app/monitoring/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "predictions.jsonl"

    def export(self, input_data: dict, prediction: dict, metadata: dict = None):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_data,
            "prediction": prediction,
            "metadata": metadata or {}
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
