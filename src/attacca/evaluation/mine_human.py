#!/usr/bin/env python
"""Human-play UI components: a pyglet window that shows the live POV while keyboard and mouse input drive the agent, and the raw POV recorder."""
import os, sys, time
from collections import defaultdict
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(REPO) + "/src")
sys.path.insert(0, os.path.join(REPO, "src/attacca/evaluation"))

from attacca.evaluation.mine_scene import CleanPOVRec
from attacca.worlds.human_deferred_labels import HUMAN_PIXEL_MARKER_CONTRACT
from attacca.worlds.human_deferred_labels import RisingEdgeLatch
from attacca.worlds.human_deferred_labels import window_to_raw_pixel


class HumanRawPOVRec(CleanPOVRec):

    def grab(self, note=""):
        import cv2
        raw9 = np.asarray(self.w.info["pov"])
        if raw9.shape != (360, 640, 3):
            raise RuntimeError(
                f"human raw POV shape changed: {raw9.shape}")
        self.frames.append(cv2.cvtColor(
            raw9, cv2.COLOR_RGB2BGR).copy())


class HumanUI:
    def __init__(self, scale=2, mouse_sens=0.15, *, caption=None,
                 window_x=None, window_y=None, prevent_initial_focus=True):
        import pyglet
        from pyglet.window import key, mouse
        self.pyglet, self.key, self.mouse = pyglet, key, mouse
        self.K2A = {
            "attack": mouse.LEFT, "use": mouse.RIGHT,
            "forward": key.W, "back": key.S, "left": key.A, "right": key.D,
            "jump": key.SPACE, "sneak": key.LSHIFT, "sprint": key.LCTRL,
            "inventory": key.E,
            **{f"hotbar.{i}": getattr(key, f"_{i}") for i in range(1, 10)},
        }
        self.W, self.H = 640 * scale, 360 * scale
        self.INFO_H = 90
        self.prevent_initial_focus = bool(prevent_initial_focus)
        self._background_window_target = (
            (int(window_x), int(window_y))
            if (self.prevent_initial_focus
                and window_x is not None and window_y is not None)
            else None)
        self._background_window_enforce_until = 0.0
        self.win = pyglet.window.Window(
            width=self.W, height=self.H + self.INFO_H,
            caption=(str(caption) if caption else "HumanPlay"),
            vsync=False, visible=not self.prevent_initial_focus)
        if self.prevent_initial_focus:
            self._map_without_focus(
                self.win,
                window_x=(None if window_x is None else int(window_x)),
                window_y=(None if window_y is None else int(window_y)))
            self._background_window_enforce_until = time.monotonic() + 2.0
        elif window_x is not None and window_y is not None:
            self.win.set_location(int(window_x), int(window_y))
        self.pressed = defaultdict(bool)
        self.mouse_delta = [0.0, 0.0]
        self.sens = mouse_sens
        self.capture = True
        self.chosen_marker = RisingEdgeLatch()
        self.resume_marker = RisingEdgeLatch()
        self.pause_marker = RisingEdgeLatch()
        self.target_selection_active = False
        self.target_selection_clickable = False
        self.ready_gate_active = False
        self.target_selection_result = None
        self.mouse_position = None
        self.abort = False
        self.abort_armed_until = 0.0
        self.quit = False
        w = self.win

        @w.event
        def on_key_press(sym, mod):
            if sym == key.ESCAPE:
                now9 = time.monotonic()
                if now9 <= float(self.abort_armed_until):
                    self.abort = True
                    self.abort_armed_until = 0.0
                else:
                    self.abort_armed_until = now9 + 3.0
                    if (not self.ready_gate_active
                            and not self.target_selection_active):
                        self.pause_marker.press()
                return pyglet.event.EVENT_HANDLED
            if sym == key.C:
                self.chosen_marker.press()
                return pyglet.event.EVENT_HANDLED
            if (sym == key.P and not self.ready_gate_active
                    and not self.target_selection_active):
                self.pause_marker.press()
                return pyglet.event.EVENT_HANDLED
            if sym == key.V and (
                    self.target_selection_active or self.ready_gate_active):
                self.abort_armed_until = 0.0
                self.resume_marker.press()
                return pyglet.event.EVENT_HANDLED
            if sym == key.TAB:
                if self.target_selection_active:
                    return pyglet.event.EVENT_HANDLED
                self.capture = not self.capture
                w.set_exclusive_mouse(self.capture)
                w.set_mouse_visible(not self.capture)
                return pyglet.event.EVENT_HANDLED
            self.pressed[sym] = True

        @w.event
        def on_key_release(sym, mod):
            if sym == key.ESCAPE:
                self.pause_marker.release()
            if sym == key.C:
                self.chosen_marker.release()
            if sym == key.P:
                self.pause_marker.release()
            if sym == key.V:
                self.resume_marker.release()
            self.pressed[sym] = False

        @w.event
        def on_mouse_press(x, y, btn, mod):
            self.mouse_position = (int(x), int(y))
            if self.target_selection_active:
                if btn == mouse.LEFT and self.target_selection_clickable:
                    pixel9 = window_to_raw_pixel(
                        x, y, viewport_x=0, viewport_y=self.INFO_H,
                        viewport_width=self.W, viewport_height=self.H,
                        raw_width=640, raw_height=360)
                    self.target_selection_result = {
                        "version": 1,
                        "contract": HUMAN_PIXEL_MARKER_CONTRACT,
                        "window_xy": [int(x), int(y)],
                        "pixel_xy": (None if pixel9 is None else
                                     [int(pixel9[0]), int(pixel9[1])]),
                        "inside_raw_rgb": bool(pixel9 is not None),
                    }
                return pyglet.event.EVENT_HANDLED
            self.pressed[btn] = True

        @w.event
        def on_mouse_release(x, y, btn, mod):
            if self.target_selection_active:
                self.pressed[btn] = False
                return pyglet.event.EVENT_HANDLED
            self.pressed[btn] = False

        @w.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_position = (int(x), int(y))
            if self.capture and not self.target_selection_active:
                self.mouse_delta[0] -= dy * self.sens
                self.mouse_delta[1] += dx * self.sens

        @w.event
        def on_mouse_drag(x, y, dx, dy, btns, mod):
            on_mouse_motion(x, y, dx, dy)

        @w.event
        def on_close():
            self.quit = True

        w.dispatch_events(); w.switch_to(); w.flip()

    @staticmethod
    def _map_without_focus(window, *, window_x=None, window_y=None):
        if not all(hasattr(window, name9) for name9 in (
                "_x_display", "_window", "_mapped", "_visible")):
            window.set_visible(True)
            if window_x is not None and window_y is not None:
                window.set_location(int(window_x), int(window_y))
            return
        from ctypes import POINTER, c_ubyte, c_ulong, cast, pointer
        from pyglet.libs.x11 import xlib

        display9 = window._x_display
        xid9 = window._window
        user_time_atom9 = xlib.XInternAtom(
            display9, b"_NET_WM_USER_TIME", False)
        cardinal_atom9 = xlib.XInternAtom(display9, b"CARDINAL", False)
        zero9 = c_ulong(0)
        xlib.XChangeProperty(
            display9, xid9, user_time_atom9, cardinal_atom9, 32,
            xlib.PropModeReplace,
            cast(pointer(zero9), POINTER(c_ubyte)), 1)
        xlib.XSelectInput(display9, xid9, xlib.StructureNotifyMask)
        xlib.XMapWindow(display9, xid9)
        event9 = xlib.XEvent()
        while True:
            xlib.XNextEvent(display9, event9)
            if event9.type == xlib.ConfigureNotify:
                window._width = event9.xconfigure.width
                window._height = event9.xconfigure.height
            elif event9.type == xlib.MapNotify:
                break
        xlib.XSelectInput(display9, xid9, window._default_event_mask)
        xlib.XFlush(display9)
        window._mapped = True
        window._visible = True
        window._update_view_size()
        window.dispatch_event("on_resize", window._width, window._height)
        window.dispatch_event("on_show")
        window.dispatch_event("on_expose")
        HumanUI._position_and_lower_x11(
            window, window_x=window_x, window_y=window_y)

    @staticmethod
    def _position_and_lower_x11(window, *, window_x=None, window_y=None):
        if not all(hasattr(window, name9) for name9 in (
                "_x_display", "_window", "_mapped")):
            if window_x is not None and window_y is not None:
                window.set_location(int(window_x), int(window_y))
            return
        from ctypes import POINTER, byref, c_uint, cast
        from pyglet.libs.x11 import xlib

        display9 = window._x_display
        xid9 = window._window
        root9 = xlib.Window()
        parent9 = xlib.Window()
        children9 = POINTER(xlib.Window)()
        child_count9 = c_uint()
        outer9 = xid9
        for _depth9 in range(8):
            ok9 = xlib.XQueryTree(
                display9, outer9, byref(root9), byref(parent9),
                byref(children9), byref(child_count9))
            if children9:
                xlib.XFree(children9)
                children9 = POINTER(xlib.Window)()
            if not ok9 or not parent9.value or parent9.value == root9.value:
                break
            outer9 = parent9.value

        if window_x is not None and window_y is not None:
            xlib.XMoveWindow(
                display9, outer9, int(window_x), int(window_y))
            moveresize_atom9 = xlib.XInternAtom(
                display9, b"_NET_MOVERESIZE_WINDOW", False)
            moveresize9 = xlib.XEvent()
            moveresize9.xclient.type = xlib.ClientMessage
            moveresize9.xclient.message_type = moveresize_atom9
            moveresize9.xclient.display = cast(
                display9, POINTER(xlib.Display))
            moveresize9.xclient.window = xid9
            moveresize9.xclient.format = 32
            moveresize9.xclient.data.l[0] = (
                xlib.NorthWestGravity | (1 << 8) | (1 << 9) | (1 << 12))
            moveresize9.xclient.data.l[1] = int(window_x)
            moveresize9.xclient.data.l[2] = int(window_y)
            moveresize9.xclient.data.l[3] = 0
            moveresize9.xclient.data.l[4] = 0
            xlib.XSendEvent(
                display9, root9, False,
                xlib.SubstructureRedirectMask | xlib.SubstructureNotifyMask,
                byref(moveresize9))

        xlib.XLowerWindow(display9, outer9)
        restack_atom9 = xlib.XInternAtom(
            display9, b"_NET_RESTACK_WINDOW", False)
        restack9 = xlib.XEvent()
        restack9.xclient.type = xlib.ClientMessage
        restack9.xclient.message_type = restack_atom9
        restack9.xclient.display = cast(display9, POINTER(xlib.Display))
        restack9.xclient.window = xid9
        restack9.xclient.format = 32
        restack9.xclient.data.l[0] = 1
        restack9.xclient.data.l[1] = 0
        restack9.xclient.data.l[2] = xlib.Below
        xlib.XSendEvent(
            display9, root9, False,
            xlib.SubstructureRedirectMask | xlib.SubstructureNotifyMask,
            byref(restack9))
        xlib.XSync(display9, False)

    def grab_mouse(self, on=True):
        self.capture = on
        self.win.set_exclusive_mouse(on)
        self.win.set_mouse_visible(not on)

    def pump(self):
        self.win.dispatch_events()
        if (self._background_window_target is not None
                and time.monotonic()
                < float(self._background_window_enforce_until)):
            self._position_and_lower_x11(
                self.win,
                window_x=self._background_window_target[0],
                window_y=self._background_window_target[1])

    def poll_action(self, noop):
        a = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in noop.items()}
        for name, k in self.K2A.items():
            if name in a:
                a[name] = int(bool(self.pressed[k]))
        dp = float(np.clip(self.mouse_delta[0], -90, 90))
        dy = float(np.clip(self.mouse_delta[1], -90, 90))
        self.mouse_delta = [0.0, 0.0]
        a["camera"] = np.array([dp, dy], dtype=np.float32)
        return a

    def consume_chosen_marker(self):
        return self.chosen_marker.consume()

    def consume_resume_marker(self):
        return self.resume_marker.consume()

    def consume_pause_marker(self):
        return self.pause_marker.consume()

    def abort_confirmation_active(self):
        active9 = time.monotonic() <= float(self.abort_armed_until)
        if not active9:
            self.abort_armed_until = 0.0
        return active9

    @staticmethod
    def _selection_overlay_rgb(
            pov_rgb, selection_overlay, candidate_overlays=None):
        import cv2
        out9 = np.asarray(pov_rgb, dtype=np.uint8).copy()

        def draw_one9(overlay9, color9, thickness9):
            proof9 = (overlay9.get("proof")
                      if isinstance(overlay9, dict) else None)
            runs9 = (proof9.get("certified_pixel_runs_yx")
                     if isinstance(proof9, dict) else None)
            if not isinstance(runs9, list):
                return
            mask9 = np.zeros(out9.shape[:2], np.uint8)
            for raw9 in runs9:
                if not isinstance(raw9, (list, tuple)) or len(raw9) != 3:
                    continue
                yy9, x09, x19 = (int(q9) for q9 in raw9)
                if 0 <= yy9 < mask9.shape[0]:
                    x09 = max(0, min(mask9.shape[1] - 1, x09))
                    x19 = max(0, min(mask9.shape[1] - 1, x19))
                    if x09 <= x19:
                        mask9[yy9, x09:x19 + 1] = 255
            contours9, _hierarchy9 = cv2.findContours(
                mask9, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours9:
                cv2.drawContours(
                    out9, contours9, -1, color9, thickness9,
                    lineType=cv2.LINE_8)

        for candidate9 in candidate_overlays or ():
            draw_one9(candidate9, (70, 255, 90), 1)
        draw_one9(selection_overlay, (0, 255, 255), 2)
        return out9

    def wait_for_start(self, pov_rgb, mission, *, start_allowed=None):
        frozen9 = np.asarray(pov_rgb).copy()
        if frozen9.shape != (360, 640, 3):
            raise ValueError(f"ready gate requires 640x360 RGB, got {frozen9.shape}")
        started9 = time.monotonic()
        self.ready_gate_active = True
        self.pressed.clear()
        self.mouse_delta = [0.0, 0.0]
        self.consume_resume_marker()
        self.grab_mouse(False)
        try:
            while not self.abort and not self.quit:
                self.pump()
                allowed9 = bool(
                    True if start_allowed is None else start_allowed())
                self.draw(
                    frozen9, mission,
                    ("READY — press V to start recording (no frames yet)"
                     if allowed9 else
                     "STAGED — waiting for this window's play permit"),
                    selection_prompt=True, ready_prompt=True,
                    freeze_reason="ready")
                if self.consume_resume_marker() and allowed9:
                    return {
                        "started": True,
                        "wait_s": max(0.0, time.monotonic() - started9),
                        "simulator_steps": 0,
                        "frames_recorded": 0,
                    }
                time.sleep(1.0 / 120.0)
            return {
                "started": False,
                "wait_s": max(0.0, time.monotonic() - started9),
                "simulator_steps": 0,
                "frames_recorded": 0,
                "reason": "quit" if self.quit else "aborted",
            }
        finally:
            self.ready_gate_active = False
            self.pressed.clear()
            self.mouse_delta = [0.0, 0.0]
            if not self.quit and not self.abort:
                self.grab_mouse(True)
                self.pump()
                self.mouse_delta = [0.0, 0.0]

    def freeze_select_target(self, pov_rgb, mission, *, reason,
                             resolve_pixel=None, allow_empty=True,
                             candidate_overlays=None,
                             selection_enabled=True):
        frozen9 = np.asarray(pov_rgb).copy()
        if frozen9.shape != (360, 640, 3):
            raise ValueError(
                f"target marker requires 640x360 RGB, got {frozen9.shape}")
        started9 = time.monotonic()
        self.target_selection_active = True
        self.target_selection_clickable = bool(selection_enabled)
        self.target_selection_result = None
        self.pressed.clear()
        self.mouse_delta = [0.0, 0.0]
        self.consume_resume_marker()
        self.grab_mouse(False)
        accepted9 = None
        overlay9 = None
        attempts9 = []
        status9 = ("PAUSED — press V to resume (no frames or sim steps)"
                   if reason == "manual_pause" else
                   "click a target, verify its exact outline, then press V"
                   if reason == "c_press" else
                   "TARGET VISIBLE 3 FRAMES — click a target; V is locked"
                   if reason == "visible_streak" else
                   "COLLECTION COMPLETE — press V to save now"
                   if reason == "quota" else
                   "click the next target; V is locked"
                   if not bool(allow_empty) else
                   "no visible target — press V to explore uncommitted")
        try:
            while not self.abort and not self.quit:
                self.pump()
                click9 = self.target_selection_result
                self.target_selection_result = None
                if click9 is not None and selection_enabled:
                    audit9 = {
                        "window_xy": click9.get("window_xy"),
                        "pixel_xy": click9.get("pixel_xy"),
                        "inside_raw_rgb": bool(click9.get("inside_raw_rgb")),
                    }
                    try:
                        if click9.get("pixel_xy") is None:
                            raise ValueError("click is outside the raw RGB viewport")
                        resolved9 = ({"marker": dict(click9), "overlay": None}
                                     if resolve_pixel is None else
                                     resolve_pixel(dict(click9)))
                        if not isinstance(resolved9, dict):
                            raise ValueError("target resolver returned no label")
                        accepted9 = dict(resolved9.get("marker") or click9)
                        overlay9 = resolved9.get("overlay")
                        audit9.update(
                            accepted=True,
                            resolved_instance_id=accepted9.get(
                                "resolved_instance_id"))
                        kind9 = str((overlay9 or {}).get("kind") or "target")
                        distance9 = (overlay9 or {}).get("distance")
                        distance_text9 = (
                            "?" if distance9 is None else f"{float(distance9):.2f} blocks")
                        status9 = (
                            f"attempt {len(attempts9) + 1}: SELECTED {kind9} | "
                            f"distance {distance_text9} | "
                            "click again to correct, V to resume")
                    except Exception as exc9:
                        accepted9 = None
                        overlay9 = None
                        audit9.update(
                            accepted=False,
                            reason=f"{type(exc9).__name__}: {exc9}")
                        reason_text9 = str(exc9).replace("\n", " ")
                        status9 = (
                            f"attempt {len(attempts9) + 1}: NOT SELECTED — "
                            f"{reason_text9[:110]}; click a GREEN outline")
                    attempts9.append(audit9)
                self.draw(
                    frozen9, mission, status9, selection_prompt=True,
                    selection_overlay=overlay9,
                    selection_cursor_xy=(
                        None if accepted9 is None else
                        accepted9.get("window_xy")),
                    candidate_overlays=(candidate_overlays
                                        if selection_enabled else None),
                    freeze_reason=str(reason),
                    flash=("QUOTA REACHED — PRESS V"
                           if reason == "quota" else None),
                    flash_color=(255, 210, 40, 255))
                if self.consume_resume_marker():
                    if accepted9 is not None or bool(allow_empty):
                        break
                    status9 = (
                        "TARGET IS VISIBLE — click inside a GREEN outline; "
                        "V cannot resume uncommitted")
                time.sleep(1.0 / 120.0)
        finally:
            self.target_selection_active = False
            self.target_selection_clickable = False
            self.target_selection_result = None
            self.pressed.clear()
            self.mouse_delta = [0.0, 0.0]
            if not self.quit:
                self.grab_mouse(True)
                self.pump()
                self.mouse_delta = [0.0, 0.0]
        if accepted9 is not None:
            accepted9["selection_attempts"] = attempts9
            accepted9["freeze_reason"] = str(reason)
        return accepted9, max(0.0, time.monotonic() - started9), attempts9

    def _label(self, text, x, y, size=13, color=(255, 255, 255, 255), anchor_x="left", bold=False):
        self.pyglet.text.Label(text, font_size=size, x=x, y=y, anchor_x=anchor_x,
                               anchor_y="center", color=color, bold=bold).draw()

    def draw(self, pov_rgb, mission, sub, flash=None, flash_color=(0, 220, 0, 255),
             selection_prompt=False, selection_overlay=None,
             candidate_overlays=None, selection_cursor_xy=None,
             freeze_reason=None, ready_prompt=False):
        import cv2
        w = self.win
        w.switch_to(); w.clear()
        display_rgb9 = self._selection_overlay_rgb(
            pov_rgb, selection_overlay, candidate_overlays
        ) if selection_prompt else np.asarray(pov_rgb)
        arr = cv2.resize(display_rgb9, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        img = self.pyglet.image.ImageData(arr.shape[1], arr.shape[0], "RGB",
                                          arr.tobytes(), pitch=arr.shape[1] * -3)
        img.blit(0, self.INFO_H)
        if (selection_prompt and isinstance(selection_cursor_xy, (list, tuple))
                and len(selection_cursor_xy) == 2):
            self._label(
                "+", int(selection_cursor_xy[0]), int(selection_cursor_xy[1]),
                size=22, anchor_x="center", color=(255, 220, 40, 255),
                bold=True)
        self._label(mission, self.W // 2, self.INFO_H - 22, size=17, anchor_x="center",
                    color=(255, 230, 90, 255), bold=True)
        self._label(sub, self.W // 2, self.INFO_H - 52, size=11, anchor_x="center",
                    color=((255, 190, 40, 255) if selection_prompt else
                           (200, 200, 200, 255)), bold=bool(selection_prompt))
        controls9 = ("READY | V start recording | ESC abort before start"
                     if ready_prompt else
                     "FROZEN (quota) | V save now | ESC abort"
                     if selection_prompt and freeze_reason == "quota" else
                     f"FROZEN ({freeze_reason or 'selection'}) | "
                     "LMB select/reselect | V resume | ESC abort"
                     if selection_prompt else
                     "WASD+mouse | C freeze+select | LMB/RMB interact | TAB mouse | ESC abort")
        abort_armed9 = self.abort_confirmation_active()
        if abort_armed9:
            controls9 = "PAUSED — ESC again within 3s to abort | V cancel/resume"
        self._label(controls9,
                    self.W // 2, 14, size=9, anchor_x="center", color=(130, 130, 130, 255))
        flash9 = "PRESS ESC AGAIN TO ABORT" if abort_armed9 else flash
        flash_color9 = ((255, 70, 70, 255) if abort_armed9 else flash_color)
        if flash9:
            self._label(flash9, self.W // 2, self.INFO_H + self.H // 2, size=34,
                        anchor_x="center", color=flash_color9, bold=True)
        w.flip()

    def play_completion_sound(self):
        try:
            tone9 = self.pyglet.media.synthesis.Sine(
                duration=0.22, frequency=880)
            tone9.play()
        except Exception:
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:
                pass

    def close(self):
        try:
            self.win.close()
        except Exception:
            pass


def ser_action(a):
    out = {}
    for k, v in a.items():
        if k in ("voxels", "mobs", "chat"):
            continue
        if k == "camera":
            v = np.asarray(v).reshape(-1)
            out[k] = [round(float(v[0]), 3), round(float(v[1]), 3)]
        else:
            try:
                out[k] = int(np.asarray(v).reshape(-1)[0])
            except (TypeError, ValueError):
                pass
    return out
