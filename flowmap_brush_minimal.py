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

import bmesh
import bpy
import math
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Euler
from mathutils.geometry import barycentric_transform, closest_point_on_tri

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
    # foreach_get, not np.array(img.pixels[:]) — the latter builds a Python list
    # of a million floats and takes about a second on a 512² canvas.
    h, w = img.size[1], img.size[0]
    buf = np.empty(h * w * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)


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
        """Paint with our own brush, cloned from the active one.

        A brush carrying a texture makes Blender draw a square texture preview
        that sits right on top of the cursor ring. We clone the active brush so
        every other setting keeps behaving exactly as it does now, strip the
        texture, and leave the user's own brush untouched.
        """
        ip = getattr(context.tool_settings, 'image_paint', None)
        src = getattr(ip, 'brush', None) if ip is not None else None

        brush = bpy.data.brushes.get(BRUSH_NAME)
        if brush is None:
            # Clone the active brush rather than making a fresh one: a blank
            # brush has defaults (tool, curve, pressure flags) that do not
            # necessarily paint here. Cloning keeps everything that already
            # works and we only strip the texture.
            if src is not None:
                brush = src.copy()
                brush.name = BRUSH_NAME
            else:
                # Image painting uses the TEXTURE_PAINT brush type; there is no
                # 'IMAGE_PAINT' enum entry.
                try:
                    brush = bpy.data.brushes.new(BRUSH_NAME, mode='TEXTURE_PAINT')
                except TypeError:
                    brush = bpy.data.brushes.new(BRUSH_NAME)
        try:
            brush.texture = None
            brush.mask_texture = None
        except Exception:
            pass
        # Blender's paint cursor is three stacked overlays: the ring
        # (use_cursor_overlay), the texture preview (use_primary_overlay) and the
        # mask preview (use_secondary_overlay). The texture one is drawn as a
        # square that sits on top of the ring, and clearing brush.texture does
        # not remove it — the overlay has to be switched off explicitly. We turn
        # all three off and draw our own ring instead.
        try:
            brush.use_cursor_overlay = False
            brush.use_primary_overlay = False
            brush.use_secondary_overlay = False
        except Exception:
            pass
        if ip is not None:
            try:
                ip.brush = brush
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

        # Our own brush overlay, since Blender's is switched off above
        self.cursor_screen = (event.mouse_region_x, event.mouse_region_y)
        self.adjust = None
        self.adjust_start = None
        self._draw_handle = None
        if GPU_AVAILABLE:
            self._draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
                self.draw_cursor, (context,), 'WINDOW', 'POST_PIXEL')

        context.window_manager.modal_handler_add(self)
        self.report_status(context)
        return {'RUNNING_MODAL'}

    def finish(self, context):
        self.remove_cursor()
        if getattr(context, 'workspace', None):
            context.workspace.status_text_set(None)
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

    # --- brush overlay ---
    # Blender's own overlays are switched off in ensure_brush, so this mirrors
    # them: a ring for the footprint and a fill whose opacity tracks strength.
    def report_status(self, context):
        """Live values in the status bar — the ring alone can't show numbers."""
        if not getattr(context, 'workspace', None):
            return
        msg = ("Flow Brush [Size=%d  Strength=%.2f]  LMB drag = paint  |  "
               "F+drag = size  |  Shift+F+drag = strength  |  RMB/ESC = exit"
               ) % (int(self.ups.size), self.strength)
        if self.adjust is not None:
            msg = "-- adjusting %s --" % self.adjust
        context.workspace.status_text_set(msg)

    def remove_cursor(self):
        handle = getattr(self, '_draw_handle', None)
        if handle is not None:
            try:
                bpy.types.SpaceImageEditor.draw_handler_remove(handle, 'WINDOW')
            except Exception:
                pass
            self._draw_handle = None

    def draw_cursor(self, context):
        if self.cursor_screen is None:
            return
        cx, cy = self.cursor_screen
        r = max(2.0, self.ups.size * 0.5)   # brush size is already in screen px
        steps = 64
        pts = []
        for i in range(steps + 1):
            a = 2.0 * math.pi * i / steps
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r, 0.0))
        try:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            gpu.state.blend_set('ALPHA')
        except Exception as err:
            self.note_draw_error(err)
            return
        shader.bind()

        # Fill and outline are drawn independently: if the fill batch ever fails
        # the ring must still appear, instead of the whole overlay vanishing.
        #
        # Explicit triangles rather than TRI_FAN, which rendered with a missing
        # wedge on some setups.
        try:
            tris = []
            for i in range(steps):
                tris.append((cx, cy, 0.0))
                tris.append(pts[i])
                tris.append(pts[i + 1])
            fill = batch_for_shader(shader, 'TRIS', {"pos": tris})
            # A hint, not a sticker: POST_PIXEL overlays composite linear and
            # then get the sRGB transform, so even 0.1 reads as a grey disc.
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.02 + self.strength * 0.08))
            fill.draw(shader)
        except Exception as err:
            self.note_draw_error(err)

        try:
            outline = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
            outline.draw(shader)
        except Exception as err:
            self.note_draw_error(err)

        try:
            gpu.state.blend_set('NONE')
        except Exception:
            pass

    def note_draw_error(self, err):
        """Report the first overlay failure once instead of silently drawing nothing.

        The draw handler runs every frame, so a print inside it would flood the
        console — but swallowing the error entirely means the cursor just
        disappears with no clue why.
        """
        if getattr(self, '_draw_error_reported', False):
            return
        self._draw_error_reported = True
        print("[FlowMap] brush overlay failed: %r" % (err,))

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.finish(context)

        if event.type == 'F':
            if event.value == 'PRESS' and self.adjust is None:
                self.adjust = 'Strength' if event.shift else 'Size'
                start = self.strength if self.adjust == 'Strength' else self.ups.size
                self.adjust_start = (event.mouse_x, start)
            elif event.value == 'RELEASE':
                self.adjust = None
                self.adjust_start = None
            self.report_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

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
            self.cursor_screen = (event.mouse_region_x, event.mouse_region_y)

            # F / Shift+F held: drag horizontally to change the value, no painting
            if self.adjust is not None and self.adjust_start is not None:
                dx = event.mouse_x - self.adjust_start[0]
                if self.adjust == 'Size':
                    self.ups.size = max(1, int(self.adjust_start[1] + dx))
                else:
                    self.strength = min(1.0, max(0.01,
                                                 self.adjust_start[1] + dx * 0.005))
                self.report_status(context)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

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


# --------------------------------------------------------------- sky preview
#
# Without this the add-on is only a brush that paints direction colours, which
# is exactly what the existing flow map painters already do. A flow map exists
# to move something, so the preview puts the canvas on a hemisphere and scrolls
# a procedural grid across it — no image asset needed, and it keeps animating
# while you paint.

PREVIEW_OBJ = "FlowMapPreview"
PREVIEW_MAT = "FlowMapPreviewMat"
PREVIEW_RADIUS = 10.0
PREVIEW_GRID = 24.0      # grid cells across the dome
PREVIEW_TICK = 0.04      # timer interval in seconds
PREVIEW_SPEED = 0.5      # one grid cell of travel per 2 s at full strength

_preview_time = 0.0
_preview_timer = False


def build_preview_dome():
    """Upper hemisphere. Blender's sphere UVs are kept: v = 0 south .. 1 north."""
    obj = bpy.data.objects.get(PREVIEW_OBJ)
    if obj is not None:
        return obj
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48,
                                         radius=PREVIEW_RADIUS,
                                         location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    if obj is None:
        return None
    obj.name = PREVIEW_OBJ
    obj.data.name = PREVIEW_OBJ + "Mesh"

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    lower = [f for f in bm.faces if f.calc_center_median().z < 0.0]
    if lower:
        bmesh.ops.delete(bm, geom=lower, context='FACES')
    orphans = [v for v in bm.verts if not v.link_faces]
    if orphans:
        bmesh.ops.delete(bm, geom=orphans, context='VERTS')
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def build_preview_material(image):
    """A grid scrolled by the flow map. Emission only, so no lights are needed."""
    mat = bpy.data.materials.get(PREVIEW_MAT)
    if mat is None:
        mat = bpy.data.materials.new(PREVIEW_MAT)
    mat.use_nodes = True
    try:
        mat.use_backface_culling = False      # visible from inside the dome
    except Exception:
        pass

    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links
    nodes.clear()

    def add(kind, name):
        n = nodes.new(kind)
        n.name = name
        n.label = name
        return n

    def val(node, index, value):
        node.inputs[index].default_value = value

    out = add("ShaderNodeOutputMaterial", "Output")
    emit = add("ShaderNodeEmission", "Emission")
    links.new(emit.outputs[0], out.inputs[0])

    # --- UV ---
    # The hemisphere sits in v = 0.5..1 of the sphere UVs. Rescale it to 0..1 so
    # the whole canvas lands on the dome: v = 0 horizon, v = 1 zenith.
    texco = add("ShaderNodeTexCoord", "UV")
    sep = add("ShaderNodeSeparateXYZ", "SepUV")
    links.new(texco.outputs["UV"], sep.inputs[0])
    vsub = add("ShaderNodeMath", "VSub")
    vsub.operation = 'SUBTRACT'
    val(vsub, 1, 0.5)
    links.new(sep.outputs[1], vsub.inputs[0])        # UV.y is output 1
    vmul = add("ShaderNodeMath", "VMul")
    vmul.operation = 'MULTIPLY'
    val(vmul, 1, 2.0)
    links.new(vsub.outputs[0], vmul.inputs[0])
    comb = add("ShaderNodeCombineXYZ", "DomeUV")
    links.new(sep.outputs[0], comb.inputs[0])
    links.new(vmul.outputs[0], comb.inputs[1])
    dome_uv = comb.outputs[0]

    # --- flow vector ---
    # The canvas is an sRGB image, so sampling linearises it (128 -> 0.216).
    # Undo with pow(1/2.2) so neutral lands back on 0.5 — otherwise an unpainted
    # sky would drift in one direction instead of standing still.
    flow_tex = add("ShaderNodeTexImage", "FlowMap")
    flow_tex.image = image
    links.new(dome_uv, flow_tex.inputs[0])

    srgb = add("ShaderNodeSeparateRGB", "SplitFlow")
    links.new(flow_tex.outputs["Color"], srgb.inputs[0])
    pow_r = add("ShaderNodeMath", "DecodeR")
    pow_r.operation = 'POWER'
    val(pow_r, 1, 1.0 / 2.2)
    links.new(srgb.outputs[0], pow_r.inputs[0])
    pow_g = add("ShaderNodeMath", "DecodeG")
    pow_g.operation = 'POWER'
    val(pow_g, 1, 1.0 / 2.2)
    links.new(srgb.outputs[1], pow_g.inputs[0])
    decoded = add("ShaderNodeCombineXYZ", "Decoded")
    links.new(pow_r.outputs[0], decoded.inputs[0])
    links.new(pow_g.outputs[0], decoded.inputs[1])

    bias = add("ShaderNodeVectorMath", "FlowBias")
    bias.operation = 'SUBTRACT'
    val(bias, 1, (0.5, 0.5, 0.5))
    links.new(decoded.outputs[0], bias.inputs[0])
    # Vector Math keeps a fourth 'Scale' float socket at index 3 — not index 1,
    # which is a vector.
    flow = add("ShaderNodeVectorMath", "Flow")
    flow.operation = 'SCALE'
    val(flow, 3, 2.0)
    links.new(bias.outputs[0], flow.inputs[0])

    # --- time ---
    time = add("ShaderNodeValue", "Time")
    time.outputs[0].default_value = _preview_time
    phase = add("ShaderNodeMath", "Phase")
    phase.operation = 'FRACT'
    links.new(time.outputs[0], phase.inputs[0])

    # --- scrolling ---
    # A single layer, because the pattern is periodic: shifting the grid by a
    # whole cell looks exactly like not shifting it, so offsetting by frac(time)
    # loops seamlessly with no reset to hide.
    #
    # The usual two-layer cross-fade (offsets t and t+0.5, blended) is meant for
    # distorting a texture where the wrap would show. Here it is actively wrong:
    # it returns to the unshifted image twice per cycle, so the sky breathes in
    # place instead of travelling in the painted direction.
    base = add("ShaderNodeVectorMath", "GridUV")
    base.operation = 'SCALE'
    val(base, 3, PREVIEW_GRID)
    links.new(dome_uv, base.inputs[0])

    off = add("ShaderNodeVectorMath", "Offset")
    off.operation = 'SCALE'
    links.new(flow.outputs[0], off.inputs[0])
    links.new(phase.outputs[0], off.inputs[3])       # 'Scale' socket

    scrolled = add("ShaderNodeVectorMath", "ScrollUV")
    scrolled.operation = 'ADD'
    links.new(base.outputs[0], scrolled.inputs[0])
    links.new(off.outputs[0], scrolled.inputs[1])

    def grid(name, uv_socket):
        """1 on a grid line, 0 midway between two lines."""
        fr = add("ShaderNodeVectorMath", name + "Frac")
        fr.operation = 'FRACTION'
        links.new(uv_socket, fr.inputs[0])
        sub = add("ShaderNodeVectorMath", name + "Sub")
        sub.operation = 'SUBTRACT'
        val(sub, 1, (0.5, 0.5, 0.5))
        links.new(fr.outputs[0], sub.inputs[0])
        ab = add("ShaderNodeVectorMath", name + "Abs")
        ab.operation = 'ABSOLUTE'
        links.new(sub.outputs[0], ab.inputs[0])
        s2 = add("ShaderNodeSeparateXYZ", name + "Sep")
        links.new(ab.outputs[0], s2.inputs[0])
        m = add("ShaderNodeMath", name + "Min")
        m.operation = 'MINIMUM'
        links.new(s2.outputs[0], m.inputs[0])
        links.new(s2.outputs[1], m.inputs[1])
        div = add("ShaderNodeMath", name + "Width")
        div.operation = 'DIVIDE'
        val(div, 1, 0.09)          # line half-width, in grid cells
        links.new(m.outputs[0], div.inputs[0])
        cl = add("ShaderNodeMath", name + "Clamp")
        cl.operation = 'MINIMUM'
        val(cl, 1, 1.0)
        links.new(div.outputs[0], cl.inputs[0])
        inv = add("ShaderNodeMath", name + "Line")
        inv.operation = 'SUBTRACT'
        val(inv, 0, 1.0)
        links.new(cl.outputs[0], inv.inputs[1])
        return inv.outputs[0]

    g = grid("Grid", scrolled.outputs[0])

    tint = add("ShaderNodeMixRGB", "Tint")
    links.new(g, tint.inputs["Fac"])
    val(tint, 1, (0.015, 0.02, 0.035, 1.0))
    val(tint, 2, (0.55, 0.85, 1.0, 1.0))
    links.new(tint.outputs[0], emit.inputs[0])
    return mat


def _preview_tick():
    """Advance the shader's clock; Blender materials do not animate themselves."""
    global _preview_time, _preview_timer
    mat = bpy.data.materials.get(PREVIEW_MAT)
    node = mat.node_tree.nodes.get("Time") if mat and mat.use_nodes else None
    if node is None:
        _preview_timer = False
        return None
    _preview_time += PREVIEW_TICK * PREVIEW_SPEED
    node.outputs[0].default_value = _preview_time
    try:
        mat.update_tag()
    except Exception:
        pass
    return PREVIEW_TICK


def start_preview_timer():
    global _preview_timer
    if _preview_timer:
        return
    _preview_timer = True
    bpy.app.timers.register(_preview_tick, first_interval=PREVIEW_TICK)


def stop_preview_timer():
    global _preview_timer
    _preview_timer = False
    try:
        bpy.app.timers.unregister(_preview_tick)
    except Exception:
        pass


class FLOW_OT_preview_build(bpy.types.Operator):
    """Show the canvas as a sky dome with the flow scrolling a grid across it"""
    bl_idname = "flow.preview_build"
    bl_label = "Preview Sky Dome"
    bl_options = {'REGISTER'}

    def execute(self, context):
        image = get_or_create_canvas()
        obj = build_preview_dome()
        if obj is None:
            self.report({'ERROR'}, "无法创建预览球体")
            return {'CANCELLED'}
        mat = build_preview_material(image)
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        start_preview_timer()
        self.report({'INFO'}, "预览已创建（画布底边=地平线，顶边=天顶）")
        return {'FINISHED'}


class FLOW_OT_preview_stop(bpy.types.Operator):
    """Remove the preview dome and stop its timer"""
    bl_idname = "flow.preview_stop"
    bl_label = "Stop Preview"
    bl_options = {'REGISTER'}

    def execute(self, context):
        stop_preview_timer()
        obj = bpy.data.objects.get(PREVIEW_OBJ)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        return {'FINISHED'}


# ------------------------------------------------------------- 3D painting
#
# Paint straight on the dome: cast the camera ray, interpolate the mesh UV at
# the hit, write pixels there. The direction comes from the movement in UV
# space, not on screen, so dragging sideways near the zenith produces the bigger
# longitude change it should — on a sphere the same screen movement covers far
# more texture up there.

def draw_brush_ring(cx, cy, r, strength, report=None):
    """Ring for the footprint, faint fill for the strength."""
    if r < 1.0:
        return
    steps = 64
    pts = []
    for i in range(steps + 1):
        a = 2.0 * math.pi * i / steps
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r, 0.0))
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
    except Exception as err:
        if report is not None:
            report(err)
        return
    shader.bind()
    try:
        tris = []
        for i in range(steps):
            tris.append((cx, cy, 0.0))
            tris.append(pts[i])
            tris.append(pts[i + 1])
        fill = batch_for_shader(shader, 'TRIS', {"pos": tris})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.02 + strength * 0.08))
        fill.draw(shader)
    except Exception as err:
        if report is not None:
            report(err)
    try:
        outline = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.9))
        outline.draw(shader)
    except Exception as err:
        if report is not None:
            report(err)
    try:
        gpu.state.blend_set('NONE')
    except Exception:
        pass


def uv_on_dome(mesh, face_index, point):
    """Mesh UV at a point inside a face, by barycentric interpolation.

    Interpolating the real UVs beats deriving them from the sphere analytically:
    this way the u orientation cannot be guessed wrong.
    """
    poly = mesh.polygons[face_index]
    uv_layer = mesh.uv_layers.active
    if uv_layer is None or len(poly.loop_indices) < 3:
        return None

    def co(loop_index):
        return mesh.vertices[mesh.loops[loop_index].vertex_index].co

    loops = list(poly.loop_indices)
    best, best_dist = None, None
    for i in range(1, len(loops) - 1):                 # fan triangulation
        tri = (loops[0], loops[i], loops[i + 1])
        a, b, c = co(tri[0]), co(tri[1]), co(tri[2])
        d = (closest_point_on_tri(point, a, b, c) - point).length
        if best_dist is None or d < best_dist:
            best, best_dist = tri, d
    if best is None:
        return None
    # barycentric_transform wants the source and target in the same dimension,
    # so the 2D UVs are lifted to 3D for the call.
    uv = barycentric_transform(point, co(best[0]), co(best[1]), co(best[2]),
                               uv_layer.data[best[0]].uv.to_3d(),
                               uv_layer.data[best[1]].uv.to_3d(),
                               uv_layer.data[best[2]].uv.to_3d())
    return (uv.x, uv.y)


def dome_pick(context, obj, event):
    """Screen position -> (u, v) on the dome in canvas space, or None."""
    region = context.region
    space = getattr(context, 'space_data', None)
    rv3d = getattr(space, 'region_3d', None) if space is not None else None
    if region is None or rv3d is None:
        return None
    mx, my = event.mouse_region_x, event.mouse_region_y
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (mx, my))
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, (mx, my))
    if origin is None or direction is None:
        return None

    inv = obj.matrix_world.inverted()
    hit, loc, _normal, face_index = obj.ray_cast(inv @ origin, inv.to_3x3() @ direction)
    if not hit or face_index is None or face_index < 0:
        return None
    uv = uv_on_dome(obj.data, face_index, loc)
    if uv is None:
        return None
    # Blender's sphere UVs run v = 0 at the south pole, so the hemisphere
    # occupies 0.5..1 — the same remap the preview shader applies. Result:
    # v = 0 horizon, v = 1 zenith.
    return (uv[0], min(1.0, max(0.0, (uv[1] - 0.5) * 2.0)))


class FLOW_OT_brush_3d(bpy.types.Operator):
    """Paint the flow map by dragging on the preview dome"""
    bl_idname = "flow.brush_3d"
    bl_label = "Paint on Dome"
    bl_description = "Drag on the preview dome to paint flow straight into the map"
    bl_options = {'REGISTER'}

    strength: bpy.props.FloatProperty(name="Strength", default=0.5, min=0.01, max=1.0)
    radius: bpy.props.IntProperty(name="Radius", default=40, min=1, max=512)

    def invoke(self, context, event):
        self.image = get_or_create_canvas()
        obj = bpy.data.objects.get(PREVIEW_OBJ)
        if obj is None:
            obj = build_preview_dome()
        if obj is None:
            self.report({'ERROR'}, "没有可绘制的天空球")
            return {'CANCELLED'}
        self.obj = obj
        self.arr = read_canvas(self.image)

        self.pressing = False
        self.last_pixel = None
        self.cursor_screen = (event.mouse_region_x, event.mouse_region_y)
        self.adjust = None
        self.adjust_start = None
        self._draw_handle = None
        if GPU_AVAILABLE:
            self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_cursor, (context,), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        self.report_status(context)
        return {'RUNNING_MODAL'}

    def report_status(self, context):
        if not getattr(context, 'workspace', None):
            return
        msg = ("Flow Brush 3D [R=%d S=%.2f]  LMB drag on the dome  |  "
               "F+drag = radius  |  Shift+F+drag = strength  |  RMB/ESC = exit"
               % (self.radius, self.strength))
        if self.adjust is not None:
            msg = "-- adjusting %s --" % self.adjust
        context.workspace.status_text_set(msg)

    def draw_cursor(self, context):
        if self.cursor_screen is None:
            return
        draw_brush_ring(self.cursor_screen[0], self.cursor_screen[1],
                        float(self.radius), self.strength, self.note_draw_error)

    def note_draw_error(self, err):
        if getattr(self, '_draw_error_reported', False):
            return
        self._draw_error_reported = True
        print("[FlowMap] dome brush overlay failed: %r" % (err,))

    def remove_cursor(self):
        handle = getattr(self, '_draw_handle', None)
        if handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            except Exception:
                pass
            self._draw_handle = None

    def finish(self, context):
        self.remove_cursor()
        write_canvas(self.image, self.arr)
        if getattr(context, 'workspace', None):
            context.workspace.status_text_set(None)
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return self.finish(context)

        if event.type == 'F':
            if event.value == 'PRESS' and self.adjust is None:
                self.adjust = 'Strength' if event.shift else 'Radius'
                start = self.strength if self.adjust == 'Strength' else self.radius
                self.adjust_start = (event.mouse_x, start)
            elif event.value == 'RELEASE':
                self.adjust = None
                self.adjust_start = None
            self.report_status(context)
            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.pressing = True
                self.last_pixel = None
            elif event.value == 'RELEASE':
                self.pressing = False
                self.last_pixel = None
                write_canvas(self.image, self.arr)
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            self.cursor_screen = (event.mouse_region_x, event.mouse_region_y)

            if self.adjust is not None and self.adjust_start is not None:
                dx = event.mouse_x - self.adjust_start[0]
                if self.adjust == 'Radius':
                    self.radius = max(1, int(self.adjust_start[1] + dx))
                else:
                    self.strength = min(1.0, max(0.01,
                                                 self.adjust_start[1] + dx * 0.005))
                self.report_status(context)
                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            if self.pressing:
                uv = dome_pick(context, self.obj, event)
                if uv is not None:
                    w, h = self.image.size[0], self.image.size[1]
                    px = (uv[0] * w, uv[1] * h)
                    if self.last_pixel is not None:
                        raw_du = px[0] - self.last_pixel[0]
                        du = raw_du
                        # the u direction wraps at the back of the dome
                        if du > w * 0.5:
                            du -= w
                        elif du < -w * 0.5:
                            du += w
                        dv = px[1] - self.last_pixel[1]
                        p1 = (self.last_pixel[0] + du, self.last_pixel[1] + dv)
                        shifts = [0.0]
                        if abs(du - raw_du) > 1e-6:      # straddled the seam
                            shifts.append(-w if raw_du > 0 else w)
                        for s in shifts:
                            stroke(self.arr,
                                   (self.last_pixel[0] + s, self.last_pixel[1]),
                                   (p1[0] + s, p1[1]),
                                   self.radius, self.strength)
                        write_canvas(self.image, self.arr)
                    self.last_pixel = px

            if context.area:
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}


# ------------------------------------------------------- viewing helpers
#
# The panel lives in the image editor sidebar, so anything that touches the 3D
# view has to go find it first.

def find_view3d(context):
    """(area, space) of the first open 3D view, or (None, None)."""
    screen = getattr(context, 'screen', None)
    if screen is None:
        return None, None
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        space = area.spaces.active
        if space is not None and getattr(space, 'region_3d', None) is not None:
            return area, space
    return None, None


class FLOW_OT_preview_camera(bpy.types.Operator):
    """Aim the 3D view at the dome — orbit it from outside, or look around inside"""
    bl_idname = "flow.preview_camera"
    bl_label = "Preview Camera"
    bl_options = {'REGISTER'}

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=(('OUTSIDE', "Frame Dome", "Orbit the dome from outside"),
               ('INSIDE', "Camera Inside", "Look around from the centre")),
        default='OUTSIDE')

    def execute(self, context):
        obj = bpy.data.objects.get(PREVIEW_OBJ)
        if obj is None:
            obj = build_preview_dome()
        if obj is None:
            self.report({'ERROR'}, "没有天空球")
            return {'CANCELLED'}
        area, space = find_view3d(context)
        if space is None:
            self.report({'ERROR'}, "没有打开的 3D 视口")
            return {'CANCELLED'}

        rv3d = space.region_3d
        # Orbiting rotates around view_location, so putting it at the centre is
        # what makes the usual middle-drag behave like the web preview.
        rv3d.view_location = obj.matrix_world.translation
        rv3d.view_perspective = 'PERSP'
        if self.mode == 'INSIDE':
            # Close to the pivot: the dome surrounds you and orbiting reads as
            # looking around rather than circling the object.
            rv3d.view_distance = 1.0
            rv3d.view_rotation = Euler((math.pi, 0.0, 0.0), 'XYZ').to_quaternion()
        else:
            rv3d.view_distance = PREVIEW_RADIUS * 3.0
            rv3d.view_rotation = Euler((math.radians(25.0), 0.0, 0.0), 'XYZ').to_quaternion()
        area.tag_redraw()
        return {'FINISHED'}


class FLOW_OT_preview_flat(bpy.types.Operator):
    """Show the dome unlit, so only the flow map is visible"""
    bl_idname = "flow.preview_flat"
    bl_label = "Flat Texture View"
    bl_options = {'REGISTER'}

    def execute(self, context):
        area, space = find_view3d(context)
        if space is None:
            self.report({'ERROR'}, "没有打开的 3D 视口")
            return {'CANCELLED'}
        shading = space.shading
        # SOLID shows the object colour rather than the material, which is why
        # the dome looks like a plain grey cap until this is switched.
        shading.type = 'MATERIAL'
        shading.light = 'FLAT'          # no studio light, no shading to fight
        shading.color_type = 'TEXTURE'
        shading.use_scene_lights = False
        shading.use_scene_world = False
        shading.show_specular_highlight = False
        area.tag_redraw()
        return {'FINISHED'}


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
        layout.separator()
        layout.label(text="Preview")
        layout.operator("flow.preview_build", icon='WORLD')
        op3d = layout.operator("flow.brush_3d", icon='BRUSH_DATA')
        op3d.strength = context.scene.flow_brush_strength

        row = layout.row(align=True)
        row.operator("flow.preview_camera", text="Frame Dome").mode = 'OUTSIDE'
        row.operator("flow.preview_camera", text="Inside").mode = 'INSIDE'
        layout.operator("flow.preview_flat", icon='SHADING_TEXTURE')

        layout.operator("flow.preview_stop", icon='CANCEL')


classes = (FLOW_OT_new_canvas, FLOW_OT_brush, FLOW_OT_brush_native,
           FLOW_OT_preview_build, FLOW_OT_preview_stop, FLOW_OT_brush_3d,
           FLOW_OT_preview_camera, FLOW_OT_preview_flat, FLOW_PT_panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.flow_brush_radius = bpy.props.IntProperty(
        name="Radius", default=40, min=1, max=512)
    bpy.types.Scene.flow_brush_strength = bpy.props.FloatProperty(
        name="Strength", default=0.5, min=0.0, max=1.0)


def unregister():
    stop_preview_timer()
    del bpy.types.Scene.flow_brush_radius
    del bpy.types.Scene.flow_brush_strength
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
