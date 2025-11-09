import cv2
import numpy as np
import math

# Global variables for mouse callback
points = []
pixels_per_mm = None

def mouse_callback(event, x, y, flags, param):
    global points
    img = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red dot
        
        if len(points) == 2:
            cv2.line(img, points[0], points[1], (0, 255, 0), 2)  # Green line
            pixel_dist = math.sqrt((points[1][0] - points[0][0])**2 + 
                                 (points[1][1] - points[0][1])**2)
            global pixels_per_mm
            pixels_per_mm = pixel_dist / 10.0  # Assuming 10mm between points
            print(f"\nCalibration complete! Scale: {pixels_per_mm:.2f} pixels/mm")
        
        cv2.imshow('Calibration', img)

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    global points, pixels_per_mm
    points = []  # Reset points
    pixels_per_mm = None
    
    # Make a clean copy for calibration
    calib_image = img.copy()
    
    # Create window and set up mouse callback
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback, [calib_image])
    
    print("\nCalibration: Please click two points that are 10mm apart on your ruler...")
    cv2.imshow('Calibration', calib_image)
    
    # Wait for user to select two points
    while len(points) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            cv2.destroyAllWindows()
            return
    
    # Wait a bit to show the measurement
    cv2.waitKey(1000)
    cv2.destroyWindow('Calibration')
    
    # Create windows with normal size
    cv2.namedWindow('Original Analysis', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edge Analysis', cv2.WINDOW_NORMAL)

    # Create copies for output
    original_output = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Dilate edges to make them more visible
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert edges to BGR for visualization
    edge_output = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

    # Process each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours
            # Draw contour
            cv2.drawContours(edge_output, [contour], -1, (255, 0, 0), 2)
            cv2.drawContours(original_output, [contour], -1, (255, 0, 0), 2)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to millimeters
            width_mm = w / pixels_per_mm
            height_mm = h / pixels_per_mm
            
            # Calculate centroid for text placement
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Draw rectangles
                cv2.rectangle(edge_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(original_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add measurements
                size_text = f"{width_mm:.1f}x{height_mm:.1f}mm"
                
                # White text with black outline on original
                cv2.putText(original_output, size_text, (cX - 40, cY), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(original_output, size_text, (cX - 40, cY), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Red text on edge image
                cv2.putText(edge_output, size_text, (cX - 40, cY), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show results
    cv2.imshow('Original Analysis', original_output)
    cv2.imshow('Edge Analysis', edge_output)
    
    print("\nProcessing complete. Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "sample1.png"  # Update this with your image path
    try:
        process_image(image_path)
    except Exception as e:
        print(f"Error: {e}")