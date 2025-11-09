import cv2
import numpy as np

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # Create a copy for the original image analysis
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
    
    # Find contours from the edge-detected image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert edges to BGR for visualization and drawing
    edge_output = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

    # Process each contour and draw on both images
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Filter out very small contours
        if area > 100:  # Adjust this threshold based on your needs
            # Draw contours
            cv2.drawContours(edge_output, [contour], -1, (255, 0, 0), 2)  # Blue on edge image
            cv2.drawContours(original_output, [contour], -1, (255, 0, 0), 2)  # Blue on original

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Get bounding rectangle dimensions
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw bounding rectangles
                cv2.rectangle(edge_output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green on edge image
                cv2.rectangle(original_output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green on original
                
                # Add size labels
                size_text = f"{w}x{h}px"
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your image path
    image_path = "sample2.png"  # You'll need to update this
    try:
        process_image(image_path)
    except Exception as e:
        print(f"Error: {e}")
