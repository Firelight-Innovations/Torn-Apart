"""Service modules for the Fire Editor daemon.

Each service registers JSON-RPC methods onto the shared
:class:`~fire_editor.rpc.Dispatcher` and may push binary frames / notifications
via the :class:`~fire_editor.server.EditorServer`. Populated from Phase E1
onward (chunks, scene, edit, texture, model). Empty in Phase E0.
"""
