# Fire Editor Protocol

> **Single source of truth: [`schema.json`](./schema.json).** This document is a
> human-readable companion. Both language bindings
> (`editor/fire_editor/_generated.py`, `editor/extension/src/protocol/generated.ts`)
> are produced by [`codegen.py`](./codegen.py). After editing `schema.json`, run
> `python editor/protocol/codegen.py` and commit all three files together. Any
> incompatible change bumps `protocol_version` (EDITOR_PRD hard rule 6).

## Transport
- WebSocket on `127.0.0.1:<port>` (developer tool; localhost only).
- The daemon picks an OS-assigned port with `--port 0` and announces it on
  **stdout** as `{"event":"listening","port":N}`. Logs go to **stderr**.

## Control channel — JSON-RPC 2.0 (text frames)
- Request: `{"jsonrpc":"2.0","id":<n>,"method":<str>,"params":<obj>}`
- Response: `{"jsonrpc":"2.0","id":<n>,"result":<obj>}` or
  `{"jsonrpc":"2.0","id":<n>,"error":{"code":<int>,"message":<str>,"data":<any?>}}`
- Notification (no `id`): `{"jsonrpc":"2.0","method":<str>,"params":<obj>}`

### Error codes
| Name | Code | Meaning |
|---|---|---|
| `PARSE_ERROR` | -32700 | malformed JSON |
| `INVALID_REQUEST` | -32600 | not a JSON-RPC 2.0 message |
| `METHOD_NOT_FOUND` | -32601 | unknown method |
| `INVALID_PARAMS` | -32602 | bad params for a known method |
| `INTERNAL_ERROR` | -32603 | unhandled handler exception |
| `VERSION_MISMATCH` | -32000 | `hello` protocol_version mismatch |
| `APP_ERROR` | -32001 | application-level failure |

## Binary channel (binary frames, same socket)
Layout, **little-endian**:

```
[u32 magic = 0x46495245][u32 schema_id][u32 payload_id][payload bytes...]
```

- `schema_id` ∈ `SchemaId`: `MESH = 1`, `TEXTURE = 2`.
- `payload_id` correlates the frame with the JSON-RPC message that announced it.
- Used for mesh buffers (positions/normals/colors/uvs/indices) and texture
  payloads (RGBA8 + header). Never base64 through JSON (hard rule 5).

### MESH payload (after the 12-byte frame header), little-endian
```
i32 cx, i32 cy, i32 cz        # chunk coord (self-describing routing)
u32 vertex_count N
u32 index_count  M
f32[N*3] positions            # absolute world meters
f32[N*3] normals              # flat per-face
f32[N*4] colors               # RGBA, greyscale x baked sunlight
f32[N*2] uvs                  # planar, tile @ 1 m
u32[M]   indices
```
Codec: `editor/fire_editor/meshcodec.py` (encode) ↔
`editor/extension/src/protocol/meshPayload.ts` (decode). Positions are absolute
world meters, so the viewport attaches every chunk at the origin.

## Methods (protocol_version 2)
| Method | Params | Result |
|---|---|---|
| `hello` | `protocol_version:int, client:str` | `ok:bool, protocol_version:int, engine_version:str, daemon_version:str` |
| `ping` | — | `pong:bool` |
| `world.open` | `seed?:int, save_path?:str` (exactly one) | `ok:bool, seed:int, config:any, edited_chunks:int` |
| `world.save` | `path:str` | `ok:bool, path:str, edited_chunks:int` |
| `chunks.set_center` | `x:float, y:float, z:float, radius?:int` | `ok:bool, requested:int` |
| `scene.stats` | — | `chunks_loaded:int, meshed:int, vertices:int, triangles:int` |
| `terrain.raycast` | `ox,oy,oz,dx,dy,dz:float, max_distance?:float` | `hit:any` (null or `{point,normal,voxel,chunk,distance}`) |

`hello` must be the first call; on `protocol_version` mismatch the daemon returns
`VERSION_MISMATCH` and the extension prompts a rebuild.

## Notifications
| Notification | Params | Meaning |
|---|---|---|
| `log` | `level:str, message:str` | daemon log line for the extension output channel |
| `chunk.ready` | `cx,cy,cz,payload_id,vertices,triangles:int` | a MESH binary frame (`payload_id`) follows for this chunk |
| `chunk.unload` | `cx,cy,cz:int` | chunk left the region; client drops its geometry |
| `stream.done` | `sent:int, removed:int` | a `set_center` streaming pass finished |

*(Method/notification tables grow per phase; keep them in sync with `schema.json`.)*
