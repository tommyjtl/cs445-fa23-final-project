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


def paste_on_background(background_img, foreground_mask, center):
    """
    Pastes a foreground mask onto a background image at a specified center coordinate.
    
    :param background_img: The background image as a NumPy array.
    :param foreground_mask: The foreground mask (3 color channels) as a NumPy array.
    :param center: Tuple (x, y) representing the center where the mask should be pasted.
    :return: The combined image as a NumPy array.
    """
    # Dimensions of the foreground mask
    h_fg, w_fg, _ = foreground_mask.shape

    # Coordinates for where to place the top-left corner of the foreground mask
    top_left_x = center[0] - w_fg // 2
    top_left_y = center[1] - h_fg // 2

    # Create a region of interest (ROI) on the background image where the mask will be placed
    roi = background_img[top_left_y:top_left_y + h_fg, top_left_x:top_left_x + w_fg]

    # Create a mask to identify non-black (non-transparent) pixels in the foreground mask
    mask = np.all(foreground_mask != [0, 0, 0], axis=-1)

    # Overlay the non-black pixels of the foreground mask onto the ROI
    roi[mask] = foreground_mask[mask]

    # Update the background image with the modified ROI
    background_img[top_left_y:top_left_y + h_fg, top_left_x:top_left_x + w_fg] = roi

    return background_img

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


'''
From CS 445
'''

from skimage import draw
import numpy as np
import matplotlib.pyplot as plt

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def specify_bottom_center(img):
    print("If it doesn't get you to the drawing mode, then rerun this function again. Also, make sure the object fill fit into the background image. Otherwise it will crash")
    # fig = plt.figure()
    # plt.imshow(img, cmap='gray')
    # fig.set_label('Choose target bottom-center location')
    # plt.axis('off')
    target_loc = np.zeros(2, dtype=int)

    def on_mouse_pressed(event):
        target_loc[0] = int(event.xdata)
        target_loc[1] = int(event.ydata)
        
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    return target_loc

def align_source(object_img, mask, background_img, bottom_center):
    ys, xs = np.where(mask == 1)
    (h,w,_) = object_img.shape
    y1 = x1 = 0
    y2, x2 = h, w
    object_img2 = np.zeros(background_img.shape)
    yind = np.arange(y1,y2)
    yind2 = yind - int(max(ys)) + bottom_center[1]
    xind = np.arange(x1,x2)
    xind2 = xind - int(round(np.mean(xs))) + bottom_center[0]

    ys = ys - int(max(ys)) + bottom_center[1]
    xs = xs - int(round(np.mean(xs))) + bottom_center[0]
    mask2 = np.zeros(background_img.shape[:2], dtype=bool)
    for i in range(len(xs)):
        mask2[int(ys[i]), int(xs[i])] = True
    for i in range(len(yind)):
        for j in range(len(xind)):
            object_img2[yind2[i], xind2[j], :] = object_img[yind[i], xind[j], :]
    mask3 = np.zeros([mask2.shape[0], mask2.shape[1], 3])
    for i in range(3):
        mask3[:,:, i] = mask2
    background_img  = object_img2 * mask3 + (1-mask3) * background_img
    # plt.figure()
    # plt.imshow(background_img)
    return object_img2, mask2

def upper_left_background_rc(object_mask, bottom_center):
    """ 
      Returns upper-left (row,col) coordinate in background image that corresponds to (0,0) in the object image
      object_mask: foreground mask in object image
      bottom_center: bottom-center (x=col, y=row) position of foreground object in background image
    """
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    upper_left_row = bottom_center[1]-int(max(ys)) 
    upper_left_col = bottom_center[0] - int(round(np.mean(xs)))
    return [upper_left_row, upper_left_col]

def crop_object_img(object_img, object_mask):
    ys, xs = np.where(object_mask == 1)
    (h,w) = object_mask.shape[:2]
    x1 = min(xs)-1
    x2 = max(xs)+2
    y1 = min(ys)-1
    y2 = max(ys)+2
    object_mask = object_mask[y1:y2, x1:x2]
    object_img = object_img[y1:y2, x1:x2, :]
    return object_img, object_mask

def get_combined_img(bg_img, object_img, object_mask, bg_ul):
    combined_img = bg_img.copy()
    (nr, nc) = object_img.shape[:2]

    for b in np.arange(object_img.shape[2]):
      combined_patch = combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b]
      combined_patch = combined_patch*(1-object_mask) + object_img[:,:,b]*object_mask
      combined_img[bg_ul[0]:bg_ul[0]+nr, bg_ul[1]:bg_ul[1]+nc, b] =  combined_patch

    return combined_img 


def specify_mask(img):
    # get mask
    print("If it doesn't get you to the drawing mode, then rerun this function again.")
    # fig = plt.figure()
    # fig.set_label('Draw polygon around source object')
    # plt.axis('off')
    # plt.imshow(img, cmap='gray')
    xs = []
    ys = []
    clicked = []

    def on_mouse_pressed(event):
        x = event.xdata
        y = event.ydata
        xs.append(x)
        ys.append(y)
        plt.plot(x, y, 'r+')

    def onclose(event):
        clicked.append(xs)
        clicked.append(ys)
    # Create an hard reference to the callback not to be cleared by the garbage
    # collector
    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)
    fig.canvas.mpl_connect('close_event', onclose)
    return clicked

def get_mask(ys, xs, img):
    mask = poly2mask(ys, xs, img.shape[:2]).astype(int)
    # fig = plt.figure()
    # plt.imshow(mask, cmap='gray')
    return mask