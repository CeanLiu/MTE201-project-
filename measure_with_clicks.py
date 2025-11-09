import cv2
import numpy as np
import math

# Controls:
#  - Left-click (calib only): mark point (max 2)
#  - Mouse wheel: zoom in/out (under cursor)
#  - Right-drag: pan
#  - u (calib only): undo last point
#  - r: reset view
#  - Enter/Space/ESC: close/confirm window

# ------------------------------
# Generic zoom/pan viewer (no loupe)
# ------------------------------
class ZoomViewer:
    def __init__(self, img, window_name="Viewer", view_w=1200, view_h=800):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.win = window_name
        self.view_w, self.view_h = view_w, view_h

        # zoom state
        self.min_zoom = min(view_w / self.W, view_h / self.H) if self.W and self.H else 1.0
        self.zoom = max(1.0, self.min_zoom)
        self.max_zoom = 20.0
        self.offset_x = (self.W - self.view_w / self.zoom) / 2.0
        self.offset_y = (self.H - self.view_h / self.zoom) / 2.0

        # interaction
        self.dragging = False
        self.drag_start = (0, 0)
        self.offset_start = (self.offset_x, self.offset_y)
        self.mouse_pos = (self.view_w // 2, self.view_h // 2)

        # HUD text (set to None to disable in subclasses)
        self.hud_text = "Wheel: zoom  |  Right-drag: pan  |  r: reset  |  Enter/Space/ESC: close"

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.view_w, self.view_h)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def disp_to_img(self, mx, my):
        ix = self.offset_x + mx / self.zoom
        iy = self.offset_y + my / self.zoom
        return ix, iy

    def _clamp_offset(self):
        # Allow slight overscroll beyond edges
        pad = 1000
        self.offset_x = max(-pad, min(self.offset_x, self.W + pad))
        self.offset_y = max(-pad, min(self.offset_y, self.H + pad))

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
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
            factor = 1.25 if delta > 0 else 1/1.25
            old_zoom = self.zoom
            new_zoom = max(self.min_zoom, min(self.zoom * factor, self.max_zoom))
            if abs(new_zoom - old_zoom) < 1e-6:
                return

            # Keep cursor's image point fixed while zooming
            ix, iy = self.disp_to_img(x, y)
            self.zoom = new_zoom
            self.offset_x = ix - x / self.zoom
            self.offset_y = iy - y / self.zoom
            self._clamp_offset()

    def _render(self):
        roi_w = self.view_w / self.zoom
        roi_h = self.view_h / self.zoom
        x0 = int(np.floor(self.offset_x))
        y0 = int(np.floor(self.offset_y))
        x1 = int(np.ceil(self.offset_y + roi_h))  # bug trap (won't use)
        x1 = int(np.ceil(self.offset_x + roi_w))
        y1 = int(np.ceil(self.offset_y + roi_h))

        canvas = np.zeros((self.view_h, self.view_w, 3), dtype=np.uint8)

        sx0 = max(0, x0); sy0 = max(0, y0)
        sx1 = min(self.W, x1); sy1 = min(self.H, y1)
        if sx1 > sx0 and sy1 > sy0:
            sub = self.img[sy0:sy1, sx0:sx1]
            dx0 = int((sx0 - self.offset_x) * self.zoom)
            dy0 = int((sy0 - self.offset_y) * self.zoom)
            dW = int(sub.shape[1] * self.zoom)
            dH = int(sub.shape[0] * self.zoom)
            sub_resized = cv2.resize(sub, (dW, dH), interpolation=cv2.INTER_LINEAR)

            # Clip to canvas bounds
            x_start = max(0, dx0); y_start = max(0, dy0)
            x_end = min(self.view_w, dx0 + dW); y_end = min(self.view_h, dy0 + dH)
            sx_start = x_start - dx0; sy_start = y_start - dy0
            sx_end = sx_start + (x_end - x_start); sy_end = sy_start + (y_end - y_start)
            canvas[y_start:y_end, x_start:x_end] = sub_resized[sy_start:sy_end, sx_start:sx_end]

        # Draw HUD if enabled
        if self.hud_text:
            cv2.putText(canvas, self.hud_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return canvas

    def run(self):
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(10) & 0xFF
            if key in (13, 32, 27):  # Enter, Space, ESC
                break
            elif key == ord('r'):
                self.zoom = max(1.0, self.min_zoom)
                self.offset_x = (self.W - self.view_w / self.zoom) / 2.0
                self.offset_y = (self.H - self.view_h / self.zoom) / 2.0
        cv2.destroyWindow(self.win)


# ------------------------------
# Zoomable calibration (with points) — max 2 clicks, no HUD overlap
# ------------------------------
class ZoomCalibrator(ZoomViewer):
    def __init__(self, img, window_name="Calibration", view_w=1200, view_h=800):
        super().__init__(img, window_name, view_w, view_h)
        self.points = []
        self.done = False
        self.canceled = False

        # Disable base HUD to avoid overlap; we'll draw our own
        self.hud_text = None

        # Enable left-click marking
        cv2.setMouseCallback(self.win, self._on_mouse_calib)

    def _on_mouse_calib(self, event, x, y, flags, param=None):
        # inherit zoom/pan behavior
        self._on_mouse(event, x, y, flags, param)

        # add point on left-click (but stop after two)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 2:
                return  # ignore extra clicks once 2 points are set
            ix, iy = self.disp_to_img(x, y)
            ix = max(0, min(self.W - 1, ix))
            iy = max(0, min(self.H - 1, iy))
            self.points.append((ix, iy))

    def _render(self):
        canvas = super()._render()
        # draw points & line in display coords
        for idx, (ix, iy) in enumerate(self.points):
            mx = int((ix - self.offset_x) * self.zoom)
            my = int((iy - self.offset_y) * self.zoom)
            if 0 <= mx < self.view_w and 0 <= my < self.view_h:
                cv2.circle(canvas, (mx, my), 5, (0, 0, 255), -1)
                cv2.putText(canvas, f"P{idx+1}", (mx + 6, my - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(canvas, f"P{idx+1}", (mx + 6, my - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if len(self.points) == 2:
            (ix1, iy1), (ix2, iy2) = self.points
            mx1 = int((ix1 - self.offset_x) * self.zoom)
            my1 = int((iy1 - self.offset_y) * self.zoom)
            mx2 = int((ix2 - self.offset_x) * self.zoom)
            my2 = int((iy2 - self.offset_y) * self.zoom)
            cv2.line(canvas, (mx1, my1), (mx2, my2), (0, 255, 0), 2)

        # Calibration HUD only (no overlap with base viewer)
        hud = "Left-click: mark (max 2)  |  Wheel: zoom  |  Right-drag: pan  |  u: undo  |  r: reset  |  Enter/Space: confirm  |  ESC: cancel"
        cv2.putText(canvas, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return canvas

    def run(self):
        while True:
            cv2.imshow(self.win, self._render())
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


# ------------------------------
# Your pipeline
# ------------------------------
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # --- Zoomable calibration ---
    calib = ZoomCalibrator(img, window_name="Calibration", view_w=1200, view_h=800)
    ok, pts = calib.run()
    if not ok:
        print("Calibration canceled.")
        return
    (x1, y1), (x2, y2) = pts
    pixel_dist = math.hypot(x2 - x1, y2 - y1)
    pixels_per_mm = pixel_dist / 10.0
    print(f"\nCalibration complete! Scale: {pixels_per_mm:.2f} pixels/mm")

    # --- Analysis (unchanged) ---
    original_output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_output = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(edge_output, [contour], -1, (255, 0, 0), 2)
            cv2.drawContours(original_output, [contour], -1, (255, 0, 0), 2)

            x, y, w, h = cv2.boundingRect(contour)
            width_mm  = w / pixels_per_mm
            height_mm = h / pixels_per_mm

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.rectangle(edge_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(original_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                size_text = f"{width_mm:.1f}x{height_mm:.1f}mm"
                cv2.putText(original_output, size_text, (cX - 40, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(original_output, size_text, (cX - 40, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(edge_output, size_text, (cX - 40, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- Zoomable viewers for both outputs ---
    ZoomViewer(original_output, "Original Analysis (Zoomable)").run()
    ZoomViewer(edge_output, "Edge Analysis (Zoomable)").run()

    print("\nDone.")

if __name__ == "__main__":
    image_path = "sample1.png"  # Update this with your image path
    try:
        process_image(image_path)
    except Exception as e:
        print(f"Error: {e}")
