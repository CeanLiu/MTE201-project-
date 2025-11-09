import cv2
import numpy as np

def process_image(image_path):
    # --- Configuration ---
    # YOU MUST SET THIS: Measure the known width of your reference object in mm.
    # In your photo, the ruler clearly shows 100mm (from 0 to 100).
    KNOWN_REFERENCE_WIDTH_MM = 100.0  # <--- NEW
    
    # This is a heuristic to find the ruler. We assume it's
    # at least 5 times wider than it is tall. Adjust if needed.
    RULER_ASPECT_RATIO_THRESHOLD = 5.0 # <--- NEW
    # ---------------------

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
        
    # Print original image dimensions
    height, width = img.shape[:2]
    print(f"Original image dimensions: {width}x{height} pixels")

    # Create a copy for the original image analysis
    original_output = img.copy()

    # Create windows with normal size
    cv2.namedWindow('Original Analysis', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edge Analysis', cv2.WINDOW_NORMAL)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Dilate edges to make them more visible
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours from the edge-detected image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert edges to BGR for visualization and drawing
    edge_output = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

    pixels_per_mm = None # <--- NEW
    
    # --- STEP 1: Find the ruler and calculate the pixels_per_mm ratio ---
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Filter out very small contours
        if area > 100:
            # Get bounding rectangle dimensions
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h # <--- NEW
            
            # Check if this looks like our ruler
            if aspect_ratio > RULER_ASPECT_RATIO_THRESHOLD: # <--- NEW
                # This is (probably) our ruler
                pixels_per_mm = w / KNOWN_REFERENCE_WIDTH_MM # <--- NEW
                
                # Draw a special box on it (Magenta) to show it's the reference
                cv2.rectangle(original_output, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(original_output, f"REFERENCE: {KNOWN_REFERENCE_WIDTH_MM}mm = {w}px", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                print(f"Found reference object: {w}px wide")
                print(f"Calculated ratio: {pixels_per_mm:.2f} pixels/mm")
                break # <--- NEW (Stop after finding the first ruler)

    # If we didn't find the ruler, we can't continue
    if pixels_per_mm is None: # <--- NEW
        print("Error: Could not find reference object (ruler).")
        print("Try adjusting RULER_ASPECT_RATIO_THRESHOLD or edge detection parameters.")
        # Still show the images, just without measurements
        cv2.imshow('Original Analysis', original_output)
        cv2.imshow('Edge Analysis', edge_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return # <--- NEW

    # --- STEP 2: Process all other contours and measure them ---
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 100: 
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio again to *skip* drawing on the ruler
            aspect_ratio = float(w) / h
            if aspect_ratio > RULER_ASPECT_RATIO_THRESHOLD: # <--- NEW
                continue # Skip the ruler, it's already marked
                
            # Draw contours
            cv2.drawContours(edge_output, [contour], -1, (255, 0, 0), 2)  # Blue on edge image
            cv2.drawContours(original_output, [contour], -1, (255, 0, 0), 2)  # Blue on original

            # Draw bounding rectangles
            cv2.rectangle(edge_output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green on edge image
            cv2.rectangle(original_output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green on original

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # --- Calculate and Add size labels in MM ---
                width_mm = w / pixels_per_mm # <--- NEW
                height_mm = h / pixels_per_mm # <--- NEW
                
                # Updated text to show mm
                size_text = f"{width_mm:.2f} x {height_mm:.2f} mm" # <--- NEW
                
                # White text with black outline on original image for better visibility
                cv2.putText(original_output, size_text, (cX - 40, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
                cv2.putText(original_output, size_text, (cX - 40, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
                # Red text on edge image
                cv2.putText(edge_output, size_text, (cX - 40, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display images side by side
    cv2.imshow('Original Analysis', original_output)
    cv2.imshow('Edge Analysis', edge_output)
    
    # Set windows to original image size
    cv2.resizeWindow('Original Analysis', width, height)
    cv2.resizeWindow('Edge Analysis', width, height)
    
    print(f"Processing complete. Press any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use the image you uploaded
    image_path = "sample1.png" # <--- UPDATED
    try:
        process_image(image_path)
    except Exception as e:
        print(f"Error: {e}")