import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
import os

import main as demo

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)
cm = app.chunk_manager
for cy in range(31, 41):
    ch = cm.get_or_create((0, cy, 0))
    s = int((ch.materials > 0).sum())
    print(f"(0,{cy},0): solid={s}")
print("--- negative & x axis ---")
for c in [(0, -40, 0), (0, -39, 0), (40, 0, 0), (39, 0, 0), (-40, 0, 0), (0, 38, 0), (0, 39, 0)]:
    ch = cm.get_or_create(c)
    print(c, "solid=", int((ch.materials > 0).sum()))
sys.stdout.flush()
os._exit(0)
