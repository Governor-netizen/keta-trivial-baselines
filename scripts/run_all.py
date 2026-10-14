#!/usr/bin/env python3
"""Run every reproduction script in order. Total time is roughly 20-40 minutes.

    python scripts/run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["01_leak_table1.py", "02_label_quality.py", "03_table2.py",
           "04_twl_and_dem.py", "05_table3_gates.py"]

for s in SCRIPTS:
    t0 = time.time()
    print(f"\n##### {s}")
    subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, check=True)
    print(f"##### {s} finished in {time.time() - t0:.0f} s")
print("\nAll results are in results/.")
