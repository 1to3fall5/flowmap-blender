bl_info = {
    "name": "FlowMap Brush (Minimal)",
    "author": "squarechen",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "Image Editor > Sidebar > FlowMap",
    "description": "Minimal flow map brush prototype. Drag with LMB to paint flow "
                   "direction into the R/G channels; B stays 255 (fully mobile).",
    "category": "Paint",
}

import bpy
import math
import numpy as np

try:
    import gpu
    from gpu_extras.batch import batch_for_shader
    GPU_AVAILABLE = True
except Exception:      # headless runs have no GL context
    GPU_AVAILABLE = False

CANVAS_NAME = "FlowMap"
CANVAS_SIZE = 512
BRUSH_NAME = "FlowMapBrush"

# Neutral flow: R=G=128 encodes "no direction", B=255 encodes "fully mobile".
NEUTRAL = (128.0 / 255.0, 128.0 / 255.0, 1.0, 1.0)


# ---------------------------------------------------------------- canvas I/O

def get_or_create_canvas(size=CANVAS_SIZE):
    img = bpy.data.images.get(CANVAS_NAME)
    if img is None:
        img = bpy.data.images.new(CANVAS_NAME, size, size, alpha=True)
    if tuple(img.size) != (size, size):
        img.scale(size, size)
    return img


def reset_canvas(img):
    h, w = img.size[1], img.size[0]
    arr = np.empty((h, w, 4), dtype=np.float32)
    arr[...] = NEUTRAL
    img.pixels.foreach_set(arr.ravel())
    img.update()


def read_canvas(img):
    h, w = img.size[1], img.size[0]
    return np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)


def write_canvas(img, arr):
    img.pixels.foreach_set(arr.ravel())
    img.update()


# ---------------------------------------------------------------- painting

def direction_to_rgb(dx, dy, strength):
    """Encode a drag direction as a flow colour.

    dx is +right and dy is +up in *image* space (row 0 is the bottom row), while
    the flow map convention has Y growing downward — hence the negation.
    """
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (128.0 / 255.0, 128.0 / 255.0, 1.0)
    vx = dx / length
    vy = -dy / length
    rng = (127.0 / 255.0) * strength
    r = min(1.0, max(0.0, 0.5 - vx * rng))
    g = min(1.0, max(0.0, 0.5 + vy * rng))
    return (r, g, 1.0)


def stamp(arr, cx, cy, radius, rgb):
    """Blend a soft round dab. Operates only on the bounding box."""
    h, w = arr.shape[:2]
    x0 = max(0, int(cx - radius)); x1 = min(w, int(cx + radius) + 1)
    y0 = max(0, int(cy - radius)); y1 = min(h, int(cy + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    sub = arr[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    a = np.clip(1.0 - d / radius, 0.0, 1.0)
    for c in range(3):
        sub[..., c] = sub[..., c] * (1.0 - a) + rgb[c] * a
    sub[..., 3] = 1.0


def stroke(arr, p0, p1, radius, strength):
    """Paint one segment from p0 to p1. Returns True if anything was drawn."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 0.5:
        return False
    rgb = direction_to_rgb(dx, dy, strength)
    steps = max(1, int(dist / max(1.0, radius * 0.25)))
    for i in range(steps + 1):
        t = i / steps
        stamp(arr, p0[0] + dx * t, p0[1] + dy * t, radius, rgb)
    return True


def region_to_pixel(context, event, image, uv_mode=None):
    """Mouse position -> image pixel coords.

    Returns (raw, pixel, uv_mode). On Blender 4.5 the Image Editor's view2d hands
    back NORMALISED UV (0..1), not pixels — which is why the first version drew
    nothing at all: a 0.1 drag never cleared the minimum-distance check. We scale
    to pixels, still detecting by magnitude in case a future release reports
    pixels instead.
    """
    region = context.region
    if region is None:
        return None, None, uv_mode

    w, h = image.size[0], image.size[1]
    view2d = getattr(region, 'view2d', None)
    if view2d is not None and hasattr(view2d, 'region_to_view'):
        raw = view2d.region_to_view(event.mouse_region_x, event.mouse_region_y)
    else:
        # assume the image fills the region
        raw = (event.mouse_region_x / max(1, region.width) * w,
               event.mouse_region_y / max(1, region.height) * h)

    if uv_mode is None:
        uv_mode = abs(raw[0]) <= 2.0 and abs(raw[1]) <= 2.0
    px = (raw[0] * w, raw[1] * h) if uv_mode else raw
    return raw, px, uv_mode


# ---------------------------------------------------------------- operators

class FLOW_OT_new_canvas(bpy.types.Operator):
    bl_idname = "flow.new_canvas"
    bl_label = "New FlowMap"
    bl_description = "Create (or reset) the flow map canvas to neutral"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        img = get_or_create_canvas()
        reset_canvas(img)
        if getattr(context, 'space_data', None) and context.space_data.type == 'IMAGE_EDITOR':
            context.space_data.image = img
        return {'FINISHED'}


class FLOW_OT_brush(bpy.types.Operator):
    bl_idname = "flow.brush"
    bl_label = "Flow Brush"
    bl_description = "LMB drag to paint flow direction; RMB or ESC to finish"
    bl_options = {'REGISTER'}

    radius: bpy.props.IntProperty(name="Radius", default=40, min=1, max=512)
    strength: bpy.props.FloatProperty(name="Strength", default=0.5, min=0.0, max=1.0)

    def invoke(self, context, event):
        self.image = get_or_create_canvas()
        if getattr(context, 'space_data', None) and context.space_data.type == 'IMAGE_EDITOR':
            context.space_data.image = self.image
        # Keep the working buffer in memory; sync to the image on release
        self.arr = read_canvas(self.image)
        self.last = None
        self.drawing = False
        self.uv_mode = None
        self.adjust = None
        self.cursor_screen = None
        self.cursor_radius_screen = 0.0
        self.cursor_raw = None      # brush centre in view2d coords
        self._draw_handle = None
        if GPU_AVAILABLE:
            self._draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
                self.draw_cursor, (context,), 'WINDOW', 'POST_PIXEL')
        # The one piece we CAN borrow from Blender: its native brush pointer
        if getattr(context, 'window', None):
            try:
                context.window.cursor_modal_set('PAINT_BRUSH')
            except Exception:
                pass
        print("[FlowMap] brush started on %s" % (tuple(self.image.size),))
        context.window_manager.modal_handler_add(self)
        self.report_status(context, None)
        return {'RUNNING_MODAL'}

    def report_status(self, context, raw):
        """Live values in the status bar — fastest way to see what's happening."""
        if not getattr(context, 'workspace', None):
            return
        msg = ("Flow Brush [R=%d S=%.2f]  LMB drag to paint  |  "
               "F+drag = radius  |  Shift+F+drag = strength") % (self.radius, self.strength)
        if self.adjust:
            msg += "   << adjusting %s >>" % self.adjust
        if raw is not None:
            msg += "   raw=(%.3f, %.3f) uv=%s draw=%s" % (
                raw[0], raw[1], self.uv_mode, self.drawing)
        context.workspace.status_text_set(msg)

    def finish(self, context):
        self.remove_cursor()
        if getattr(context, 'window', None):
            try:
                context.window.cursor_modal_restore()
            except Exception:
                pass
        write_canvas(self.image, self.arr)
        if getattr(context, 'workspace', None):
            context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}

    def remove_cursor(self):
        handle = getattr(self, '_draw_handle', None)
        if handle is not None:
            try:
                bpy.types.SpaceImageEditor.draw_handler_remove(handle, 'WINDOW')
            except Exception:
                pass
            self._draw_handle = None

    def update_cursor(self, context, event, raw=None):
        """Brush ring position/size in region pixels — follows the view zoom.

        Pass raw=None to keep the centre where it is and only recompute the
        radius. That's what sizing does: the brush grows in place while the
        mouse only supplies a delta.
        """
        region = context.region
        if raw is not None:
            self.cursor_raw = raw
        if region is None or self.cursor_raw is None:
            self.cursor_screen = None
            return
        w = self.image.size[0]
        view2d = getattr(region, 'view2d', None)
        if view2d is not None and hasattr(view2d, 'view_to_region'):
            center = view2d.view_to_region(self.cursor_raw[0], self.cursor_raw[1], clip=False)
            edge = view2d.view_to_region(self.cursor_raw[0] + self.radius / w,
                                         self.cursor_raw[1], clip=False)
            self.cursor_screen = center
            self.cursor_radius_screen = abs(edge[0] - center[0])
        else:
            self.cursor_screen = (event.mouse_region_x, event.mouse_region_y)
            self.cursor_radius_screen = self.radius / w * region.width

    def draw_cursor(self, context):
        """GPU overlay: ring for the footprint, faint fill for the strength.

        Blender draws its brush cursor inside the paint system, so there is no
        Python API to reuse it — this mirrors the look instead: an outline for
        the size and a fill whose opacity tracks strength.
        """
        if self.cursor_screen is None or self.cursor_radius_screen < 1.0:
            return
        cx, cy = self.cursor_screen
        r = self.cursor_radius_screen
        steps = 64
        ring = []
        for i in range(steps + 1):
            a = 2.0 * math.pi * i / steps
            ring.append((cx + math.cos(a) * r, cy + math.sin(a) * r, 0.0))
        try:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set('ALPHA')

            fan = [(cx, cy, 0.0)] + ring[:-1]      # TRI_FAN needs the centre first
            fill = batch_for_shader(shader, 'TRI_FAN', {"pos": fan})
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.04 + self.strength * 0.16))
            fill.draw(shader)

            outline = batch_for_shader(shader, 'LINE_STRIP', {"pos": ring})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
            outline.draw(shader)

            gpu.state.blend_set('NONE')
        except Exception:
            pass

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.finish(context)

        # Blender's own brush shortcuts: F sizes the brush, Shift+F is strength
        if event.type == 'F':
            if event.value == 'PRESS':
                self.adjust = 'strength' if event.shift else 'radius'
            elif event.value == 'RELEASE':
                self.adjust = None
            self.report_status(context, None)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.drawing = True
                _, px, self.uv_mode = region_to_pixel(context, event, self.image, self.uv_mode)
                self.last = px
            elif event.value == 'RELEASE':
                self.drawing = False
                self.last = None
                write_canvas(self.image, self.arr)
                if context.area:
                    context.area.tag_redraw()
            self.report_status(context, None)
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            raw, px, self.uv_mode = region_to_pixel(context, event, self.image, self.uv_mode)

            if self.adjust is not None:
                # accumulate from the per-event delta, like Blender's brush sizing
                delta = event.mouse_x - event.mouse_prev_x
                if self.adjust == 'radius':
                    self.radius = max(1, min(512, self.radius + int(delta * 0.5)))
                    context.scene.flow_brush_radius = self.radius
                else:
                    self.strength = max(0.0, min(1.0, self.strength + delta * 0.002))
                    context.scene.flow_brush_strength = self.strength
                self.update_cursor(context, event)   # centre stays put
                if context.region:
                    context.region.tag_redraw()
                self.report_status(context, raw)
                return {'RUNNING_MODAL'}

            self.update_cursor(context, event, raw)
            if context.region:
                context.region.tag_redraw()
            self.report_status(context, raw)
            if self.drawing and px is not None and self.last is not None:
                if stroke(self.arr, self.last, px, self.radius, self.strength):
                    write_canvas(self.image, self.arr)
                    if context.area:
                        context.area.tag_redraw()
                self.last = px
        return {'RUNNING_MODAL'}


class FLOW_OT_brush_native(bpy.types.Operator):
    """Paint a flow map using Blender's paint engine.

    Blender builds the dab (shape, hardness, pressure) — we only feed it a colour
    derived from the drag direction, recomputed before every dab.

    The stroke is driven explicitly through bpy.ops.paint.image_paint() rather
    than by passing mouse events through. Blender's own stroke handling reads the
    brush colour once per stroke, so PASS_THROUGH would paint a single flat
    colour — that is why this cannot simply delegate the drag. The technique
    follows Clemens Beute's flow_map_painter.
    """
    bl_idname = "flow.brush_native"
    bl_label = "Flow Brush (Native Paint)"
    bl_description = "Uses Blender's paint engine — native brush settings, pressure and cursor"
    bl_options = {'REGISTER'}

    # Flow strength: how far the encoded direction deviates from neutral.
    # Deliberately not unified_paint_settings.strength, which is also the dab
    # opacity — using it for both attenuates twice and low values paint nothing.
    strength: bpy.props.FloatProperty(name="Strength", default=0.5, min=0.01, max=1.0)

    def ensure_brush(self, context):
        """Use our own brush instead of whatever was last active.

        A brush carrying a texture makes Blender draw a square texture preview
        that sits right on top of the cursor ring, which is what makes the
        sizing overlay look blocked. Creating a dedicated texture-free brush
        also means we never disturb the user's own brushes.
        """
        brush = bpy.data.brushes.get(BRUSH_NAME)
        if brush is None:
            brush = bpy.data.brushes.new(BRUSH_NAME, mode='IMAGE_PAINT')
        try:
            brush.texture = None
            brush.mask_texture = None
            brush.blend = 'MIX'
            brush.use_pressure_strength = True
            brush.use_pressure_size = False
        except Exception:
            pass
        try:
            context.tool_settings.image_paint.brush = brush
        except Exception:
            pass
        return brush

    def invoke(self, context, event):
        self.image = get_or_create_canvas()
        space = getattr(context, 'space_data', None)
        if space is not None and space.type == 'IMAGE_EDITOR':
            space.image = self.image
            try:
                space.mode = 'PAINT'      # gives us Blender's own brush cursor
            except Exception:
                pass
        self.ensure_brush(context)

        # Colour-space caveat: a data texture would want 'Non-Color', but Blender
        # may then round-trip the colour through linear space and the encoded 128
        # would not survive. Verify by exporting: untouched pixels should read 128.
        try:
            self.image.colorspace_settings.name = 'sRGB'
        except Exception:
            pass

        # Unified settings are what Blender's brush panel drives, so the sidebar
        # stays live for size / strength / colour.
        ups = context.scene.tool_settings.unified_paint_settings
        ups.use_unified_color = True
        ups.use_unified_size = True
        ups.use_unified_strength = True
        self.ups = ups

        self.pressing = False
        self.last_mouse = None     # window coords — the direction comes from these
        self.last_dot = None       # where the previous dab landed
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def finish(self, context):
        try:
            self.ups.color = (0.5, 0.5, 1.0)
        except Exception:
            pass
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}

    def place_dot(self, context, mouse):
        """One dab, produced by Blender's paint engine."""
        area = context.area
        if area is None:
            return
        local = (mouse[0] - area.x, mouse[1] - area.y)
        stroke = [{
            "name": "",
            "is_start": True,
            "location": (0, 0, 0),
            "mouse": local,
            "mouse_event": local,
            "pressure": 1.0,
            "size": self.ups.size,
            "time": 1,
            "x_tilt": 0,
            "y_tilt": 0,
        }]
        try:
            bpy.ops.paint.image_paint(stroke=stroke, mode='NORMAL')
        except Exception:
            pass

    def paint_dots(self, context, mouse):
        """Space dabs along the drag, interpolating when the mouse moves fast."""
        spacing = max(2.0, self.ups.size * 0.15)
        if self.last_dot is None:
            self.place_dot(context, mouse)
            self.last_dot = mouse
            return
        dx = mouse[0] - self.last_dot[0]
        dy = mouse[1] - self.last_dot[1]
        dist = math.hypot(dx, dy)
        if dist < spacing:
            return
        steps = int(dist / spacing)
        for i in range(1, steps + 1):
            t = i / steps
            self.place_dot(context, (self.last_dot[0] + dx * t, self.last_dot[1] + dy * t))
        self.last_dot = mouse

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.finish(context)

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.pressing = True
                self.last_dot = None
            elif event.value == 'RELEASE':
                self.pressing = False
                self.last_dot = None
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            mouse = (event.mouse_x, event.mouse_y)

            if self.last_mouse is not None:
                dx = mouse[0] - self.last_mouse[0]
                dy = mouse[1] - self.last_mouse[1]
                if math.hypot(dx, dy) > 0.0:
                    # window Y grows upward, and so do image rows
                    r, g, b = direction_to_rgb(dx, dy, self.strength)
                    try:
                        self.ups.color = (r, g, b)
                    except Exception:
                        pass

            if self.pressing:
                self.paint_dots(context, mouse)

            self.last_mouse = mouse
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # let Blender handle F / Shift+F sizing and everything else
        return {'PASS_THROUGH'}


class FLOW_PT_panel(bpy.types.Panel):
    bl_label = "FlowMap"
    bl_idname = "FLOW_PT_panel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "FlowMap"

    def draw(self, context):
        # Only the native operator is exposed. The numpy one (flow.brush) is kept
        # in the code and can still be run via F3 search, but showing both was
        # just confusing.
        layout = self.layout
        layout.operator("flow.new_canvas", icon='IMAGE_DATA')
        layout.separator()
        op = layout.operator("flow.brush_native", icon='BRUSH_DATA')
        op.strength = context.scene.flow_brush_strength
        layout.prop(context.scene, "flow_brush_strength", text="Flow Strength")


classes = (FLOW_OT_new_canvas, FLOW_OT_brush, FLOW_OT_brush_native, FLOW_PT_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.flow_brush_radius = bpy.props.IntProperty(
        name="Radius", default=40, min=1, max=512)
    bpy.types.Scene.flow_brush_strength = bpy.props.FloatProperty(
        name="Strength", default=0.5, min=0.0, max=1.0)


def unregister():
    del bpy.types.Scene.flow_brush_radius
    del bpy.types.Scene.flow_brush_strength
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
