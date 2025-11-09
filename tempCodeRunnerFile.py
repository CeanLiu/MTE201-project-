import cv2
import numpy as np
import math

class ZoomCalibrator:
    def __init__(self, img, window_name="Calibration", view_w=1200, view_h=800):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.win = window_name
        self.view_w, self.view_h = view_w, view_h

        self.zoom = 1.0
        self.min_zoom = min(view_w / self.W, view_h / self.H)
        self.zoom = max(self.zoom, self.min_zoom)
        self.max_zoom = 20.0
        self.offset_x = (self.W - self.view_w / self.zoom) / 2.0
        self.offset_y = (self.H - self.view_h / self.zoom) / 2.0

        self.dragging = False
        self.drag_start = (0, 0)
        self.offset_start = (self.offset_x, self.offset_y)
        self.mouse_pos = (self.view_w // 2, self.view_h // 2)

        self.points = []
        self.done = False
        self.canceled = False

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.view_w, self.view_h)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def disp_to_img(self, mx, my):
        ix = self.offset_x + mx / self.zoom
        iy = self.offset_y + my / self.zoom
        return ix, iy

    def img_to_disp(self, ix, iy):
        mx = int((ix - self.offset_x) * self.zoom)
        my = int((iy - self.offset_y) * self.zoom)
        return mx, my

    def _clamp_offset(self):
        vw = self.view_w / self.zoom
        vh = self.view_h / self.zoom
        self.offset_x = max(-1000, min(self.offset_x, self.W + 1000))
        self.offset_y = max(-1000, min(self.offset_y, self.H + 1000))

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self.disp_to_img(x, y)
            ix = max(0, min(self.W - 1, ix))
            iy = max(0, min(self.H - 1, iy))
            self.points.append((ix, iy))

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.offset_start = (self.offset_x, self.offset_y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            self.offset_x = self.offset_start[0] - dx / self.zoom
            self.offset_y = self.offset_start[1] - dy / self.zoom
            self._clamp_offset()

        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            factor = 1.25 if delta > 0 else 0.8
            old_zoom = self.zoom
            new_zoom = max(self.min_zoom, min(self.zoom * factor, self.max_zoom))
            if abs(new_zoom - old_zoom) < 1e-6:
                return

            ix, iy = self.disp_to_img(x, y)
            self.zoom = new_zoom
            self.offset_x = ix - x / self.zoom
            self.offset_y = iy - y / self.zoom
            self._clamp_offset()

    def _render_view(self):
        roi_w = self.view_w / self.zoom
        roi_h = self.view_h / self.zoom
        x0 = int(np.floor(self.offset_x))
        y0 = int(np.floor(self.offset_y))
        x1 = int(np.ceil(self.offset_x + roi_w))
        y1 = int(np.ceil(self.offset_y + roi_h))

        canvas = np.zeros((self.view_h, self.view_w, 3), dtype=np.uint8)

        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(self.W, x1)
        sy1 = min(self.H, y1)

        if sx1 > sx0 and sy1 > sy0:
            sub = self.img[sy0:sy1, sx0:sx1]
            dx0 = int((sx0 - self.offset_x) * self.zoom)
            dy0 = int((sy0 - self.offset_y) * self.zoom)
            dW = int(sub.shape[1] * self.zoom)
            dH = int(sub.shape[0] * self.zoom)
            sub_resized = cv2.resize(sub, (dW, dH), interpolation=cv2.INTER_LINEAR)
            canvas[dy0:dy0 + dH, dx0:dx0 + dW] = sub_resized[:self.view_h - dy0, :self.view_w - dx0]

        for idx, (ix, iy) in enumerate(self.points):
            mx, my = self.img_to_disp(ix, iy)
            if 0 <= mx < self.view_w and 0 <= my < self.view_h:
                cv2.circle(canvas, (mx, my), 5, (0, 0, 255), -1)
                cv2.putText(canvas, f"P{idx + 1}", (mx + 6, my - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(canvas, f"P{idx + 1}", (mx + 6, my - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if len(self.points) >= 2:
            (ix1, iy1), (ix2, iy2) = self.points[:2]
            mx1, my1 = self.img_to_disp(ix1, iy1)
            mx2, my2 = self.img_to_disp(ix2, iy2)
            cv2.line(canvas, (mx1, my1), (mx2, my2), (0, 255, 0), 2)

        cv2.putText(canvas, "Left-click: mark  |  Right-drag: pan  |  Wheel: zoom  |  u: undo  |  r: reset  |  Enter: confirm",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return canvas

    def run(self):
        print("\nCalibration: pick TWO points 10mm apart.")
        while True:
            view = self._render_view()
            cv2.imshow(self.win, view)
            key = cv2.waitKey(10) & 0xFF

            if key in (13, 32):  # Enter or Space
                if len(self.points) >= 2:
                    self.done = True
                    break
            elif key == 27:  # ESC
                self.canceled = True
                break
            elif key == ord('u') and self.points:
                self.points.pop()
            elif key == ord('r'):
                self.zoom = max(1.0, self.min_zoom)
                self.offset_x = (self.W - self.view_w / self.zoom) / 2.0
                self.offset_y = (self.H - self.view_h / self.zoom) / 2.0

        cv2.destroyWindow(self.win)
        return self.done, self.points[:2]


# integrate with your same process_image() code as before

