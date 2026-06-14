import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import numpy as np, main as demo
import os

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
cm = app.chunk_manager
# Directly generate chunks at increasing y and count solids
for cy in [0, 1, 5, 10, 20, 30, 40, 50, 60, 100, 200, 500, 1000]:
    ch = cm.get_or_create((0, cy, 0))
    s = int((ch.materials > 0).sum())
    print(f"chunk (0,{cy},0): solid={s}")
sys.stdout.flush()
os._exit(0)
