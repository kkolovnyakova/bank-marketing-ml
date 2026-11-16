
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT / "split_data.py")], check=True, cwd=ROOT)
subprocess.run([sys.executable, str(ROOT / "src" / "prepare_data.py")], check=True, cwd=ROOT)
subprocess.run([sys.executable, str(ROOT / "src" / "train_models.py")], check=True, cwd=ROOT)
