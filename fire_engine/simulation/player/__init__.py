"""
fire_engine.simulation.player — Player API: thin human-control layer
(same interface as an NPC agent).

Session 1 exports only the free-fly camera controller.  Future sessions will
add the embodied player agent (walking, collision, inventory) using the same
component interface as NPC entities.

All exports are panda3d-free; the controller receives input via InputState
(set by App) and operates purely on Transform/Quat/Vec3.

Example
-------
    from fire_engine.simulation.player import FlyController
    from fire_engine.render  import instantiate

    camera_go = instantiate(name="MainCamera")
    ctrl = camera_go.add_component(FlyController, move_speed=10.0)
    # App calls ctrl.set_input_state(inp) each frame before registry.run_frame()
"""

from fire_engine.simulation.player.fly_controller import FlyController

__all__ = [
    "FlyController",
]
