# MTE201 Project - Pixel Measurement Tool

A simple, interactive tool to measure pixel distances between two points on any image. Perfect for analyzing photos from your phone or any digital image.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

2. **Run the program:**
   ```bash
   python pixel_measure.py
   ```

3. **Select your image** from the file dialog (supports PNG, JPG, JPEG, BMP, TIFF)

4. **Click two points** on the image to measure the pixel distance

## Usage

1. A file dialog will open - select your image file
2. The image will open in a zoomable window
3. **Left-click** to select two points on the image
4. The pixel distance between the points will be displayed in real-time
5. Use the controls below to navigate and make precise measurements

### Controls

- **Left-click**: Select measurement points (select 2 points)
- **Right-click + Drag**: Pan around the image
- **Mouse Wheel**: Zoom in/out
- **'r' key**: Reset view to default zoom and position
- **'c' key**: Clear all selected points
- **Enter/Space/ESC**: Close the program

### Features

- Zoom and pan functionality for precise point selection
- Real-time distance calculation displayed on the image
- Shows both total pixel distance and X/Y components
- Console output with detailed measurement information

## Requirements

- Python 3.6 or higher
- OpenCV (`opencv-python`)
- NumPy
- tkinter (usually included with Python on Windows/Mac, may need `python3-tk` on Linux)

### Installation

Install all required packages:
```bash
pip install opencv-python numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Output

The program displays:
- **Total pixel distance** between the two selected points
- **X and Y components** (horizontal and vertical distances)
- **Point coordinates** in the console when you close the window

## Example

After selecting two points, you'll see:
- Visual markers (red circles) at each point
- A green line connecting the points
- Distance measurement displayed on the image
- Detailed measurement information in the console