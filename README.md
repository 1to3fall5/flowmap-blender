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
3. **Flow Brush (Native Paint)** — puts the editor into Paint mode; **hold LMB and drag** to paint
4. RMB or `ESC` to finish

Because it runs inside Blender's paint mode you get the native brush cursor, and
`F` / `Shift`+`F` are Blender's own sizing keys.

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

## How it works

Blender's brushes write a *colour*, but a flow map needs a *direction vector*
that changes with every stroke. So the add-on recomputes the brush colour from
the drag direction on every mouse move and lets Blender's paint engine handle
the stroke — dab shape, hardness and pressure are Blender's own, the colour is
ours.

A numpy-based operator (`flow.brush`) that writes pixels directly is still in the
file and can be reached via F3 search for "Flow Brush", but it is not shown in
the panel.
