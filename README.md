# flowmap-blender

Flow map painting add-on for Blender 4.5.

Paints flow direction straight into a texture's R/G channels (B stays 255 = fully
mobile), so the result can be fed directly to a flow-map material — skyboxes,
water, cloth, anything that needs per-pixel motion direction.

## Install

Blender → Edit → Preferences → Add-ons → **Install** → pick
`flowmap_brush_minimal.py`, then enable it.

## Usage

1. Switch to the **Image Editor**, press `N` to open the sidebar → **FlowMap** tab
2. **New FlowMap** — create (or reset) the canvas, filled with neutral `(128,128,255)`
3. Pick a brush mode (see [Two brush modes](#two-brush-modes))
4. **Hold LMB and drag** to paint; RMB or `ESC` to finish

## Two brush modes

| | **Flow Brush** | **Flow Brush (Native Paint)** |
| :--- | :--- | :--- |
| Who draws | the add-on, with numpy | Blender's paint engine |
| Dab shape / hardness / pen pressure | fixed soft round | whatever the active brush has |
| `F` / `Shift`+`F` | implemented here | Blender's own |
| Cursor ring | custom overlay | Blender's `draw_circle_2d` |

**Strength means flow strength in both modes** — how far the encoded direction
deviates from neutral. It is deliberately kept separate from Blender's brush
strength, which only controls dab opacity; sharing one value would attenuate
twice and low values would paint almost nothing.

The native mode is the more complete one. The numpy mode is a self-contained
fallback that does not depend on the paint engine.

## Shortcuts

| Key | Action |
| :--- | :--- |
| Hold `F` + drag | Brush radius |
| Hold `Shift`+`F` + drag | Brush strength |

Same as Blender's own paint brushes.

## Conventions

- **R** = horizontal flow, **G** = vertical flow, `128/255` is neutral (no flow)
- **B** = mobility mask (`255` = flows, `0` = static)

These match the web version, [skyflow-designer](https://github.com/1to3fall5/skyflow-designer),
so flow maps can be moved between the two.

## Why the direction can't come from the brush

Blender's brushes write a *colour*, while a flow map needs a *direction vector*
that changes with every stroke. The trick is to recompute the brush colour from
the drag direction on every mouse move — Blender still builds the dab (shape,
hardness, pressure, spacing), it just never gets to choose the colour.

The technique follows Clemens Beute's
[flow_map_painter](https://github.com/ClemensBeute/flow_map_painter): write to
`tool_settings.unified_paint_settings.color` and drive
`bpy.ops.paint.image_paint()` with an explicit stroke.

The numpy mode instead writes pixels directly, bypassing the paint engine
altogether.
