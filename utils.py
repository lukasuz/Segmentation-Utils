""" Some semantic segmentation utility functions.
"""
import numpy as np

_black_white = [[0,0,0],[255,255,255]]

def one_hot_image(img, colours=_black_white):
    """ Transforms a rgb colour to a one hot version represention the colours.

    The position of the colours is relevant. Each 
    # Arguments:
        img: 3D rgb colour img.
        colours: List of colours in that image, default is binary
            black and white.
    # Returns:
        3D Matrix with len(colours) channels.
    """
    segmentation_maps = np.zeros((img.shape[0], img.shape[1], len(colours)))
    for i in range(len(colours)):
        class_segmentation_map = np.all(np.equal(img, colours[i]), axis=-1)
        segmentation_maps[:, :, i] = class_segmentation_map
    return segmentation_maps

def one_hot_image_to_label_image(one_hot_img):
    """ Transforms a one hotted segmentation image to its label representation.

    # Arguments:
        one_hot_img: Image that has been transformed into its one hot representation.
    
    # Returns:
        2D matrix where each point represents its label.
    """
    return np.argmax(one_hot_img, axis=-1)

def label_image_to_rgb_image(label_img, colours=_black_white):
    """ Transforms a label image back to a RGB image.
    
    # Arguments:
        label_img: 2D label image
        colours: List of colours in that image, default is binary
            black and white.
    # Returns:
        3D rgb colour image
    """
    return np.array(colours)[label_img.astype(int)]
