import json
from pathlib import Path


class config:
    def __init__(self, config_path: str):
        config_path = Path(config_path)
        data = json.loads(config_path.read_text())
        self.model = data["model"]
        self.data_config = data["data_config"]
        self.pipline_config = data["pipline_config"]
        self.lr_config = data["lr_config"]
        self.train_config = data["train_config"]
        self.test_config = data["test_config"]
        self.checkpoint_config = data["checkpoint_config"]
        self.log_interval = 50
        self.load_from = data["download_from"]
        self.resume_from = None
        self.debug = True
        self.seed = 42
