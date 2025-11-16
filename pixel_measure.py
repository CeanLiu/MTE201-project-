import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
import os

class PixelMeasurer:
    def __init__(self, img, window_name="Pixel Measurement", view_w=1200, view_h=800):
        self.img = img.copy()
        self.original_img = img.copy()
        self.H, self.W = img.shape[:2]
        self.win = window_name
        self.view_w, self.view_h = view_w, view_h

        # Points for measurement
        self.points = []

        # Zoom state
        self.min_zoom = min(view_w / self.W, view_h / self.H) if self.W and self.H else 1.0
        self.zoom = max(1.0, self.min_zoom)
        self.max_zoom = 20.0
        self.offset_x = (self.W - self.view_w / self.zoom) / 2.0
        self.offset_y = (self.H - view_h / self.zoom) / 2.0

        # Interaction
        self.dragging = False
        self.drag_start = (0, 0)
        self.offset_start = (self.offset_x, self.offset_y)
        self.mouse_pos = (self.view_w // 2, self.view_h // 2)

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.view_w, self.view_h)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def disp_to_img(self, mx, my):
        """Convert display coordinates to image coordinates"""
        ix = self.offset_x + mx / self.zoom
        iy = self.offset_y + my / self.zoom
        return ix, iy

    def _clamp_offset(self):
        """Keep offset within reasonable bounds"""
        pad = 1000
        self.offset_x = max(-pad, min(self.offset_x, self.W + pad))
        self.offset_y = max(-pad, min(self.offset_y, self.H + pad))

    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        # Left click: add measurement point
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 2:
                # Reset if we already have 2 points
                self.points = []
            ix, iy = self.disp_to_img(x, y)
            ix = max(0, min(self.W - 1, int(ix)))
            iy = max(0, min(self.H - 1, int(iy)))
            self.points.append((ix, iy))
            print(f"Point {len(self.points)}: ({ix}, {iy})")

        # Right click and drag: pan
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

        # Mouse wheel: zoom
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
        """Render the image with points and measurement line"""
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

            # Clip to canvas bounds
            x_start = max(0, dx0)
            y_start = max(0, dy0)
            x_end = min(self.view_w, dx0 + dW)
            y_end = min(self.view_h, dy0 + dH)
            sx_start = x_start - dx0
            sy_start = y_start - dy0
            sx_end = sx_start + (x_end - x_start)
            sy_end = sy_start + (y_end - y_start)
            canvas[y_start:y_end, x_start:x_end] = sub_resized[sy_start:sy_end, sx_start:sx_end]

        # Draw points and line
        for idx, (ix, iy) in enumerate(self.points):
            mx = int((ix - self.offset_x) * self.zoom)
            my = int((iy - self.offset_y) * self.zoom)
            if 0 <= mx < self.view_w and 0 <= my < self.view_h:
                # Draw point
                cv2.circle(canvas, (mx, my), 8, (0, 0, 255), -1)
                cv2.circle(canvas, (mx, my), 10, (255, 255, 255), 2)
                # Label point
                label = f"P{idx+1}"
                cv2.putText(canvas, label, (mx + 12, my - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(canvas, label, (mx + 12, my - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Draw line and measurement if we have 2 points
        if len(self.points) == 2:
            (ix1, iy1), (ix2, iy2) = self.points
            mx1 = int((ix1 - self.offset_x) * self.zoom)
            my1 = int((iy1 - self.offset_y) * self.zoom)
            mx2 = int((ix2 - self.offset_x) * self.zoom)
            my2 = int((iy2 - self.offset_y) * self.zoom)
            
            # Draw line
            cv2.line(canvas, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            
            # Calculate pixel distance
            pixel_distance = math.hypot(ix2 - ix1, iy2 - iy1)
            dx = abs(ix2 - ix1)
            dy = abs(iy2 - iy1)
            
            # Display measurement text at midpoint
            mid_x = (mx1 + mx2) // 2
            mid_y = (my1 + my2) // 2
            text = f"Distance: {pixel_distance:.2f} pixels"
            text2 = f"({dx} x {dy})"
            
            # Background for text
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (text_w2, text_h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            max_w = max(text_w, text_w2)
            cv2.rectangle(canvas, 
                         (mid_x - max_w//2 - 5, mid_y - text_h - text_h2 - 15),
                         (mid_x + max_w//2 + 5, mid_y + 5),
                         (0, 0, 0), -1)
            cv2.rectangle(canvas, 
                         (mid_x - max_w//2 - 5, mid_y - text_h - text_h2 - 15),
                         (mid_x + max_w//2 + 5, mid_y + 5),
                         (255, 255, 255), 1)
            
            # Text
            cv2.putText(canvas, text, (mid_x - text_w//2, mid_y - text_h2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, text2, (mid_x - text_w2//2, mid_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Draw HUD
        hud_lines = [
            "Left-click: Select points (2 points)",
            "Right-drag: Pan  |  Mouse wheel: Zoom  |  r: Reset view",
            "c: Clear points  |  Enter/Space/ESC: Close"
        ]
        y_offset = 25
        for line in hud_lines:
            cv2.putText(canvas, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(canvas, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y_offset += 25

        return canvas

    def run(self):
        """Main loop"""
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(10) & 0xFF
            
            if key in (13, 32, 27):  # Enter, Space, ESC
                break
            elif key == ord('r'):
                # Reset view
                self.zoom = max(1.0, self.min_zoom)
                self.offset_x = (self.W - self.view_w / self.zoom) / 2.0
                self.offset_y = (self.H - self.view_h / self.zoom) / 2.0
            elif key == ord('c'):
                # Clear points
                self.points = []
                print("Points cleared")
        
        cv2.destroyWindow(self.win)
        
        # Print final measurement if we have 2 points
        if len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            pixel_distance = math.hypot(x2 - x1, y2 - y1)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            true_L = pixel_distance * 0.0508 - 0.1586
            print(f"\n{'='*50}")
            print(f"Final Measurement:")
            print(f"  Point 1: ({x1}, {y1})")
            print(f"  Point 2: ({x2}, {y2})")
            print(f"  Pixel Distance: {pixel_distance:.2f} pixels")
            print(f"  Delta X: {dx} pixels")
            print(f"  Delta Y: {dy} pixels")
            print (f"  True Length: {true_L:.2f} mm (approx.) +- 0.18mm")
            print(f"{'='*50}")


def select_image_file():
    """Open file dialog to select an image"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def main():
    print("="*50)
    print("Pixel Measurement Tool")
    print("="*50)
    print("\nSelect an image file to measure...")
    
    # Select image file
    image_path = select_image_file()
    
    if not image_path:
        print("No file selected. Exiting.")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        print("Please make sure it's a valid image file.")
        return
    
    print(f"\nImage loaded: {image_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")
    print("\nInstructions:")
    print("  - Left-click to select two points")
    print("  - Right-click and drag to pan")
    print("  - Mouse wheel to zoom")
    print("  - Press 'r' to reset view")
    print("  - Press 'c' to clear points")
    print("  - Press Enter/Space/ESC to close")
    print("\n" + "="*50 + "\n")
    
    # Run the measurement tool
    measurer = PixelMeasurer(img)
    measurer.run()


if __name__ == "__main__":
    main()

