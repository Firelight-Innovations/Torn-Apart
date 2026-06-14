"""Capture the Cornell GI room from increasing camera distances to find where
the cascade-transition breaks the interior lighting."""

import math, sys
from pathlib import Path
import main as demo
from fire_engine.core.math3d import Vec3, Quat

app = demo.build_demo()
app.input_state.mouse_captured = False
app._set_mouse_capture(False)

# noon-ish clear
import tools.screenshot as ss

ss._apply_sky_settings(app, 13.0, "clear")

cx, cy, z0 = demo.build_gi_test_room(app)
cz = z0 + 2.25
print("ROOM at", (cx, cy, z0))

# warm up the room build (let cascades assemble + propagate settle)
hold = float(app._clock.game_time_of_day)


def step(n):
    for _ in range(n):
        app.taskMgr.step()
        app._clock.game_time_of_day = hold


step(120)


# Camera always looks toward the room center (down +Y) tilted slightly down.
def look_at_room(dist):
    # stand on -Y side at given distance from room center, at interior height
    pos = Vec3(cx, cy - dist, cz + 0.5)
    app.camera_go.transform.position = pos
    # yaw 0 faces +Y; small downward pitch
    app.camera_go.transform.local_rotation = (
        Quat.from_axis_angle(Vec3.RIGHT, math.radians(-6.0))
    ).normalized()


out = Path("tools/out/diag")
for dist in (4.2, 12.0, 22.0, 35.0, 60.0):
    look_at_room(dist)
    step(60)  # let cascades recenter + re-propagate after the jump
    p = out / f"room_dist_{int(dist):02d}.png"
    app.win.save_screenshot(str(p))
    print("WROTE", p, "dist=", dist)

app.lighting_pipeline.shutdown()
