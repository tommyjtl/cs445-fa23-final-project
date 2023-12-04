import cv2
import numpy as np
import json
import argparse


def load_points_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        loaded_points = [(point['x'], point['y']) for point in data['points']]
    return loaded_points


def save_points_to_json(points, json_file_path='points.json'):
    points_dict = {'points': [{'x': p[0], 'y': p[1]} for p in points]}
    with open(json_file_path, 'w') as file:
        json.dump(points_dict, file, indent=4)
    print(f"Points saved to {json_file_path}")


def mouse_event(event, x, y, flags, param):
    global points, moving_index
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 68:
            points.append((x, y))
        save_points_to_json(points)
    elif event == cv2.EVENT_MOUSEMOVE:
        if moving_index is not None:
            points[moving_index] = (x, y)
            save_points_to_json(points)
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        if moving_index is not None:
            save_points_to_json(points)
        moving_index = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, point in enumerate(points):
            if np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) < 10:
                moving_index = i
                break


# Setup argparse
parser = argparse.ArgumentParser(
    description="Add and move points on an image.")
parser.add_argument("image_path", help="Path to the input image.")
parser.add_argument(
    "--json_path", help="Path to the input JSON file with points.", default="")
args = parser.parse_args()

# Load the image
img = cv2.imread(args.image_path)
if img is None:
    raise ValueError(f"Image not found at {args.image_path}")
img_copy = img.copy()

# Initialize global variables
points = []
moving_index = None

# Optionally load points from a JSON file
if args.json_path:
    try:
        points = load_points_from_json(args.json_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")

# Create a window and set the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_event)

while True:
    img_copy = img.copy()
    for i, point in enumerate(points):
        cv2.circle(img_copy, point, 8, (0, 0, 255), -1)
        cv2.putText(
            img_copy,
            str(i+1),
            (point[0]+5, point[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Image", img_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Escape key
        break

cv2.destroyAllWindows()
