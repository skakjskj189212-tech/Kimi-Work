#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

# Resolve path relative to this script
workspace = Path(__file__).parent.resolve()
cmd = [
    sys.executable,
    str(workspace / "pinterest_qc_pro.py"),
    "--config", str(workspace / "qc_config.yaml"),
    "--archetype", str(workspace / "archetype_anchors"),
    "--input", str(workspace / "pinterest_raw"),
    "--output", str(workspace / "output")
]

print(f"Running command: {' '.join(cmd)}")
sys.stdout.flush()

# Execute subprocess and stream output directly
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(workspace))

# Print stdout to console in real-time
for line in process.stdout:
    print(line, end="")
    sys.stdout.flush()

process.wait()
sys.exit(process.returncode)
