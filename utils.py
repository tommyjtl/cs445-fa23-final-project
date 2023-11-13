import cv2
import dlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

def better_plotter(images, plot_size=(8, 4)):
    # plt.close() # Clear previous plot buffer
    fig, axes = plt.subplots(1, len(images) + 1, figsize=plot_size)
    
    plt.axis('off')  # Turn off the axis

    for i in range(0, len(images), 1):
        axes[i].imshow(images[i]['img'])
        axes[i].set_title(images[i]['title'])

# Function to get coordinates from landmarks
def get_landmark_coords(landmarks):
    coords = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)])
    return coords

# def draw_face(image, faces):
#     """
#     Draw a bounding box on the detected face
#     """
#     print(faces[0])
    
#     image_with_face = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
#     cv2.rectangle(
#         image_with_face, 
#         (faces[0].left(), faces[0].top()), 
#         (faces[0].right(), faces[0].bottom()), 
#         (255, 0, 0), 
#         5
#     )

#     return image_with_face

# def draw_landmarks(image, landmarks):
#     # Loop through each face detected
#     image_landmarks = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

#     for i in range(len(landmarks)):
#         # Draw the landmarks with circles
#         for n in range(0, 68):
#             x = landmarks[i].part(n).x
#             y = landmarks[i].part(n).y
#             cv2.circle(image_landmarks, (x, y), 6, (255, 0, 0), -1)
    
#     return image_landmarks

# def rect_contains(rect, point):
#     if point[0] < rect[0] or point[0] > rect[2] or point[1] < rect[1] or point[1] > rect[3]:
#         return False
#     return True

# def get_triangles(image, landmark_points):    
#     # Applying Delaunay triangulation
#     rect = (0, 0, image.shape[1], image.shape[0])
#     subdiv = cv2.Subdiv2D(rect)
#     subdiv.insert(landmark_points)
#     triangles = subdiv.getTriangleList()
    
#     print(len(landmark_points), len(triangles))
    
#     # Converting to int
#     triangles = np.array(triangles, dtype=np.int32)

#     return triangles, rect

# def draw_triangulation(image, landmarks):
#     image_triangulation = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
#     # Collecting the facial landmark points
#     landmark_points = []
#     for n in range(0, 68):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
#         landmark_points.append((x, y))
    
#     triangles, rect = get_triangles(image_triangulation, landmark_points)
    
#     for t in triangles:
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
        
#         # Drawing the triangles
#         if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
#             cv2.line(image_triangulation, pt1, pt2, (0, 255, 0), 2)
#             cv2.line(image_triangulation, pt2, pt3, (0, 255, 0), 2)
#             cv2.line(image_triangulation, pt3, pt1, (0, 255, 0), 2)

#     return image_triangulation

# def extract_face(image, landmarks):
#     image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
#     image_triangulation = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
#     # Creating a blank mask for the face
#     face_mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
#     # Collecting the facial landmark points
#     landmark_points = []
#     for n in range(0, 68):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
#         landmark_points.append((x, y))
    
#     triangles, rect = get_triangles(image, landmark_points)
#     # print(len(triangles))
    
#     for t in triangles:
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
#         # print(pt1, pt2, pt3)
        
#         # # Drawing the triangles
#         # if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
#         #     cv2.line(image_triangulation, pt1, pt2, (0, 255, 0), 1)
#         #     cv2.line(image_triangulation, pt2, pt3, (0, 255, 0), 1)
#         #     cv2.line(image_triangulation, pt3, pt1, (0, 255, 0), 1)
        
#         if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
#             # Draw the triangle in the mask
#             cv2.fillConvexPoly(face_mask, np.array([pt1, pt2, pt3], dtype=np.int32), 255)
    
#     # # Extracting the face pixels
#     # extracted_face_trig = cv2.bitwise_and(image_triangulation, image_triangulation, mask=face_mask)
#     # extracted_face_rgb_trig = cv2.cvtColor(extracted_face_trig, cv2.COLOR_BGR2RGB)

#     # Extracting the face pixels
#     extracted_face = cv2.bitwise_and(image, image, mask=face_mask)
#     extracted_face_rgb = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2RGB)
    
#     # Determine the bounding rectangle of the face region
#     x_coordinates = [point[0] for point in landmark_points]
#     y_coordinates = [point[1] for point in landmark_points]
#     x_min, x_max = min(x_coordinates), max(x_coordinates)
#     y_min, y_max = min(y_coordinates), max(y_coordinates)

#     # # Crop the masked image to the bounding rectangle
#     # cropped_face_trig = extracted_face_trig[y_min:y_max, x_min:x_max]
#     # cropped_face_rgb_trig = cv2.cvtColor(cropped_face_trig, cv2.COLOR_BGR2RGB)
    
#     # Crop the masked image to the bounding rectangle
#     cropped_face = extracted_face[y_min:y_max, x_min:x_max]
#     cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    
#     # Adjusting the triangle coordinates to the cropped image's origin
#     adjusted_triangles = []
#     for t in triangles:
#         pt1 = (t[0] - x_min, t[1] - y_min)
#         pt2 = (t[2] - x_min, t[3] - y_min)
#         pt3 = (t[4] - x_min, t[5] - y_min)
    
#         if rect_contains((0, 0, cropped_face.shape[1], cropped_face.shape[0]), pt1) and \
#            rect_contains((0, 0, cropped_face.shape[1], cropped_face.shape[0]), pt2) and \
#            rect_contains((0, 0, cropped_face.shape[1], cropped_face.shape[0]), pt3):
#             adjusted_triangles.append([pt1, pt2, pt3])

#     cropped_face_with_trig = cropped_face_rgb.copy()
    
#     # Optional: Draw the adjusted triangles on the cropped image for visualization
#     for t in adjusted_triangles:
#         cv2.line(cropped_face_with_trig, t[0], t[1], (0, 255, 0), 1)
#         cv2.line(cropped_face_with_trig, t[1], t[2], (0, 255, 0), 1)
#         cv2.line(cropped_face_with_trig, t[2], t[0], (0, 255, 0), 1)

#     return cropped_face_rgb, cropped_face_with_trig, adjusted_triangles
    