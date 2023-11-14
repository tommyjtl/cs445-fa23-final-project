import cv2
import dlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

def better_plotter(images, plot_size=(8, 4)):
    # plt.close() # Clear previous plot buffer
    if len(images) == 1:
        fig, axes = plt.subplots(1, len(images) + 1, figsize=plot_size)
    else:
        fig, axes = plt.subplots(1, len(images), figsize=plot_size)
    
    plt.axis('off')  # Turn off the axis

    for i in range(0, len(images), 1):
        axes[i].imshow(images[i]['img'])
        axes[i].set_title(images[i]['title'])

# Function to get coordinates from landmarks
def get_landmark_coords(landmarks):
    coords = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)])
    return coords

def crop_face_with_padding(image, face, padding_ratio=1/3):
    """
    Crop the face from the image with additional padding. If the padding exceeds the image size,
    the exceeded regions will be filled with black pixels.

    Parameters:
    - image: The source image.
    - face: The coordinates of the face (dlib.rectangle object).
    - padding_ratio: Ratio of face size to be used as padding (default: 1/3).

    Returns:
    Cropped image including the face with additional padding.
    """
    # Calculate the dimensions of the detected face
    face_width = face.right() - face.left()
    face_height = face.bottom() - face.top()

    # Calculate the padding
    padding_width = int(face_width * padding_ratio)
    padding_height = int(face_height * padding_ratio)

    # Create a new image with padding filled with black pixels
    new_width = face_width + 2 * padding_width
    new_height = face_height + 2 * padding_height
    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calculate the position to paste the cropped face in the new image
    paste_x = max(padding_width - face.left(), 0)
    paste_y = max(padding_height - face.top(), 0)

    # Adjust the coordinates for cropping, ensuring they are within the original image
    crop_left = max(face.left() - padding_width, 0)
    crop_top = max(face.top() - padding_height, 0)
    crop_right = min(face.right() + padding_width, image.shape[1])
    crop_bottom = min(face.bottom() + padding_height, image.shape[0])

    # Crop the image and paste it into the padded image
    cropped_face = image[crop_top:crop_bottom, crop_left:crop_right]
    padded_image[paste_y:paste_y + cropped_face.shape[0], paste_x:paste_x + cropped_face.shape[1]] = cropped_face

    return cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

def resize_image_to_target_height(source_img, target_img):
    """
    Resize the source image to have the same height as the target image,
    while maintaining the aspect ratio of the source image.

    Parameters:
    - source_img: The source image to be resized.
    - target_img: The target image whose height is to be matched.

    Returns:
    Resized source image.
    """
    # Calculate the aspect ratio of the source image
    aspect_ratio = source_img.shape[1] / source_img.shape[0]

    # Calculate the new width to maintain the aspect ratio
    new_height = target_img.shape[0]
    new_width = int(new_height * aspect_ratio)

    # Check if the source image is larger than the target image
    if source_img.shape[0] > target_img.shape[0] or source_img.shape[1] > target_img.shape[1]:
        # Use Gaussian blurring before resizing to avoid aliasing
        source_img = cv2.GaussianBlur(source_img, (5, 5), 0)

    # Resize the image
    resized_source_img = cv2.resize(source_img, (new_width, new_height))

    return resized_source_img

def concat_images(img1, img2):
    """
    Concatenate two images horizontally.

    Parameters:
    - img1: First image.
    - img2: Second image.

    Returns:
    Concatenated image.
    """
    # Check if the heights of the two images are the same
    if img1.shape[0] != img2.shape[0]:
        raise ValueError("The heights of both images must be the same.")

    # Concatenate the images horizontally
    concatenated_img = np.hstack((img1, img2))

    return concatenated_img