import cv2
import dlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

def better_plotter(images, plot_size=(8, 4)):
    """
    Display a list of images with their titles in a single or multiple subplot layout.

    Parameters:
    - images: A list of dictionaries, each containing an image ('img') and a title ('title').
    - plot_size: A tuple specifying the size of the plot (default is (8, 4)).

    Returns:
    None. This function displays the images using matplotlib.
    """
    # Check if there is only one image to plot
    if len(images) == 1:
        plt.figure(figsize=plot_size)
        
        # Display the image, handling both color and grayscale images
        if len(images[0]['img'].shape) > 2:
            plt.imshow(images[0]['img'])
        else:
            plt.imshow(images[0]['img'], cmap='gray')
            
        plt.title(images[0]['title'])
        plt.show()
        
    else:
        # Create subplots for multiple images
        fig, axes = plt.subplots(1, len(images), figsize=plot_size)
        
        # Iterate through each image and display it on its respective subplot
        for i in range(len(images)):
            if len(images[i]['img'].shape) > 2:
                axes[i].imshow(images[i]['img'])
            else:
                axes[i].imshow(images[i]['img'], cmap='gray')
                
            axes[i].set_title(images[i]['title'])
    

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

def crop_face_with_padding_corners(image, corners, padding_ratio=1/3):
    """
    Crop the face from the image with additional padding based on the coordinates of its corners.
    If the padding exceeds the image size, the exceeded regions will be filled with black pixels.

    Parameters:
    - image: The source image (numpy array).
    - corners: A tuple of four points (top left, top right, bottom left, bottom right) representing the bounding box of the face.
    - padding_ratio: Ratio of face size to be used as padding (default: 1/3).

    Returns:
    - Cropped image including the face with additional padding.
    """
    top_left, top_right, bottom_left, bottom_right = corners

    # Calculate face width and height
    face_width = top_right[0] - top_left[0]
    face_height = bottom_left[1] - top_left[1]

    # Calculate the padding
    padding_width = int(face_width * padding_ratio)
    padding_height = int(face_height * padding_ratio)

    # Create a new image with padding filled with black pixels
    new_width = face_width + 2 * padding_width
    new_height = face_height + 2 * padding_height
    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calculate the position to paste the cropped face in the new image
    paste_x = max(padding_width - top_left[0], 0)
    paste_y = max(padding_height - top_left[1], 0)

    # Adjust the coordinates for cropping, ensuring they are within the original image
    crop_left = max(top_left[0] - padding_width, 0)
    crop_top = max(top_left[1] - padding_height, 0)
    crop_right = min(bottom_right[0] + padding_width, image.shape[1])
    crop_bottom = min(bottom_right[1] + padding_height, image.shape[0])

    # Crop the image and paste it into the padded image
    cropped_face = image[crop_top:crop_bottom, crop_left:crop_right]
    padded_image[paste_y:paste_y + cropped_face.shape[0], paste_x:paste_x + cropped_face.shape[1]] = cropped_face

    return cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)


def is_inside_bound(point, rectangle):
    """
    Check if a point is inside the bounds of a rectangle.

    Parameters:
    - point: A tuple (x, y) representing the point.
    - rectangle: A tuple (x1, y1, x2, y2) representing the rectangle's top-left and bottom-right corners.

    Returns:
    - True if the point is inside the rectangle, False otherwise.
    """
    x, y = point
    return (rectangle[0] <= x <= rectangle[2]) and (rectangle[1] <= y <= rectangle[3])


# we are using the convex hull points but this also works given the facial landmark points not within a convex hull
def get_triagulation(img, points):
    """
    Find the Delaunay triangulation for a given set of points on an image.

    Parameters:
    - img: The source image.
    - points: A list of (x, y) tuples representing points on the image.

    Returns:
    - A list of tuples, each representing a triangle in the Delaunay triangulation.
      Each tuple contains indices of the points forming the triangle.
    """
    # Define the bounding rectangle for the image
    rectangle = (0, 0, img.shape[1], img.shape[0])
    
    # Initialize a subdivision for triangulation within the defined rectangle
    subdivision = cv2.Subdiv2D(rectangle)

    
    # Insert each point into the subdivision
    for point in points:
        subdivision.insert((int(point[0]), int(point[1])))
        
    # Initialize an empty list to store the Delaunay triangulation
    delauney_triangulation = []

    # Iterate through each triangle generated in the subdivision
    for triangle in subdivision.getTriangleList():

        # Extract vertices of the triangle
        a = (triangle[0], triangle[1])
        b = (triangle[2], triangle[3])
        c = (triangle[4], triangle[5])

        # Check if all vertices of the triangle are within the image bounds
        if is_inside_bound(a, rectangle) and is_inside_bound(b, rectangle) and is_inside_bound(c, rectangle):
            point_indices = []

            # For each vertex of the triangle, find the corresponding point index
            for i, point in enumerate(points):
                if abs(a[0] - point[0]) < 1.0 and abs(a[1] - point[1]) < 1.0:
                    point_indices.append(i)
                if abs(b[0] - point[0]) < 1.0 and abs(b[1] - point[1]) < 1.0:
                    point_indices.append(i)
                if abs(c[0] - point[0]) < 1.0 and abs(c[1] - point[1]) < 1.0:
                    point_indices.append(i)
                    
            # If all three vertices have corresponding points, add the triangle to the triangulation list
            if len(point_indices) == 3:
                delauney_triangulation.append((point_indices[0], point_indices[1], point_indices[2]))

    return delauney_triangulation

def affine_transform(src, src_triangles, dst_triangles, size):
    """
    Calculate the affine transform from one triangle to another and apply it to an image.

    Parameters:
    - src: Source image.
    - src_triangles: List of 3 tuples representing the vertices of the source triangle.
    - dst_triangles: List of 3 tuples representing the vertices of the destination triangle.
    - size: Tuple (width, height) representing the size of the output image.

    Returns:
    - Transformed image with the affine transformation applied.
    """
    
    # Given a pair of triangles, find the affine transform.
    warp_matrix = cv2.getAffineTransform(np.float32(src_triangles), np.float32(dst_triangles))

    # Return the Affine Transform found to the src image
    return cv2.warpAffine(
            src, warp_matrix, 
            (size[0], size[1]), None, 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

def morph_triangular_region(triangle_1, triangle_2, img_1, img_2):
    """
    Morph a triangular region from one image to another.

    Parameters:
    - triangle_1: List of 3 tuples representing the vertices of the triangle in the source image.
    - triangle_2: List of 3 tuples representing the vertices of the triangle in the destination image.
    - img_1: Source image.
    - img_2: Destination image.

    Returns:
    - The destination image with the morphed triangular region.
    """
    
    # Find bounding rectangles for each triangle
    r1 = cv2.boundingRect(np.float32([triangle_1]))
    r2 = cv2.boundingRect(np.float32([triangle_2]))

    # Offset points by the top left corner of the bounding rectangle
    offset_triangle_1 = [(vertex[0] - r1[0], vertex[1] - r1[1]) for vertex in triangle_1]
    offset_triangle_2 = [(vertex[0] - r2[0], vertex[1] - r2[1]) for vertex in triangle_2]

    # Apply affine transform to the source image
    transformed_img = affine_transform(img_1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]], 
                                           offset_triangle_1, offset_triangle_2, (r2[2], r2[3]))

    # Mask for the triangular region in the destination image
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(offset_triangle_2), (1.0, 1.0, 1.0), 16, 0)

    # Apply the transformation to the destination image
    img2_rect_area = img_2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2_rect_area = img2_rect_area * (1 - mask) + transformed_img * mask
    img_2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_rect_area

    return img_2

def apply_affine_transformation(delauney, hull_1, hull_2, img_1, img_2):
    """
    Applies affine transformations to morph a face from one image to another using Delaunay triangulation.

    Parameters:
    - delauney: A list of triangles, each represented by a tuple of three point indices, obtained from Delaunay triangulation.
    - hull_1: List of (x, y) tuples representing facial landmarks on the source image.
    - hull_2: List of (x, y) tuples representing facial landmarks on the destination image.
    - img_1: The source image.
    - img_2: The destination image.

    Returns:
    - The destination image with the facial features of the source image morphed onto it.
    """
    
    # Create a copy of the destination image
    img2_copy = np.copy(img_2)

    # Iterate over each triangle in the Delaunay triangulation
    for triangle in delauney:
        tri1 = [hull_1[i] for i in triangle]
        tri2 = [hull_2[i] for i in triangle]

        # Morph each triangular region from the source to the destination image
        morph_triangular_region(tri1, tri2, img_1, img2_copy)

    return img2_copy


def create_mask_from_image(image):
    """
    Create a mask from an image by turning all non-black pixels to white.

    :param image: Input image as a NumPy array (with 3 channels).
    :return: Masked image where all non-black pixels are turned to white.
    """
    # Create a boolean mask where True indicates non-black pixels
    non_black_mask = np.any(image != [0, 0, 0], axis=-1)

    # Create an empty white image
    white_image = np.ones_like(image) * 255

    # Apply the mask
    masked_image = np.where(non_black_mask[..., None], white_image, image)

    return masked_image

def overlay_mask(background, mask, center, color):
    """
    Overlay a mask onto a background image at a specified center and color the mask.

    :param background: Background image as a NumPy array.
    :param mask: Mask image as a NumPy array.
    :param center: Tuple (x, y) representing the center coordinates in the background.
    :param color: Tuple (B, G, R) representing the color to fill the mask.
    :return: Image with mask overlay.
    """
    # Dimensions of the background and mask
    bg_height, bg_width = background.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    # Calculate top-left corner from the center
    top_left = (center[0] - mask_width // 2, center[1] - mask_height // 2)

    # Handle cases where the mask might go out of the image boundaries
    start_x = max(top_left[0], 0)
    start_y = max(top_left[1], 0)
    end_x = min(start_x + mask_width, bg_width)
    end_y = min(start_y + mask_height, bg_height)

    # Overlay the mask
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            if np.all(mask[y - start_y, x - start_x] != [0, 0, 0]):
                background[y, x] = color

    return background

def paste_mask_on_canvas(background_size, mask, center):
    """
    Paste a mask onto an all-black canvas with the size of the background.

    :param background_size: Tuple (width, height) representing the size of the background.
    :param mask: Mask image as a NumPy array.
    :param center: Tuple (x, y) representing the center coordinates on the canvas.
    :return: Canvas with mask overlay.
    """
    # Create a black canvas
    canvas = np.zeros((background_size[1], background_size[0], 3), dtype=np.uint8)

    # Dimensions of the mask
    mask_height, mask_width = mask.shape[:2]

    # Calculate top-left corner from the center
    top_left_x = max(center[0] - mask_width // 2, 0)
    top_left_y = max(center[1] - mask_height // 2, 0)

    # Calculate the region on canvas where the mask will be placed
    bottom_right_x = min(top_left_x + mask_width, background_size[0])
    bottom_right_y = min(top_left_y + mask_height, background_size[1])

    # Adjust mask if it goes beyond the canvas
    mask_adjusted = mask[:(bottom_right_y - top_left_y), :(bottom_right_x - top_left_x)]

    # Paste the mask onto the canvas
    canvas[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = mask_adjusted

    return canvas

def find_bounding_box(coordinates):
    """
    Finds the bounding box for a set of coordinates.

    Parameters:
    - coordinates: A list of tuples, where each tuple represents a coordinate (x, y).

    Returns:
    - A dictionary containing the corners of the bounding box: 'top_left', 'top_right', 
      'bottom_left', and 'bottom_right'.
    """
    # Initialize the extreme points
    min_x = min(coordinates, key=lambda x: x[0])[0]
    max_x = max(coordinates, key=lambda x: x[0])[0]
    min_y = min(coordinates, key=lambda x: x[1])[1]
    max_y = max(coordinates, key=lambda x: x[1])[1]

    # Determine the corners of the bounding box
    top_left = (min_x, min_y)
    top_right = (max_x, min_y)
    bottom_left = (min_x, max_y)
    bottom_right = (max_x, max_y)

    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }