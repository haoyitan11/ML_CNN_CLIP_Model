from pathlib import Path
from datetime import datetime

def log_line(exp_dir: str, msg: str):
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(exp_dir) / "train.log"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
