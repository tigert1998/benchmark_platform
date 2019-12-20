import numpy as np
import cv2

CROPPING_FRACTION = 0.875
IMSIZE = 224


def crop_and_resize(image: np.ndarray, imsize: int):
    original_height, original_width, channels = image.shape
    assert channels == 3

    crop_size = round(min(original_height, original_width) * CROPPING_FRACTION)
    top = round((original_height - crop_size) / 2.0)
    left = round((original_width - crop_size) / 2.0)
    image = image[top:top+crop_size, left:left + crop_size, :]

    return cv2.resize(
        image, (imsize, imsize),
        interpolation=cv2.INTER_LINEAR)


def apply_mean_and_scale(image: np.ndarray, mean, scale):
    return (image - mean) * scale


def std_preprocess(image: np.ndarray, imsize: int, output_type):
    """Standard Preprocess.

    Preprocess a RGB image (np.array) in the exact same way as 
    tensorflow lite imagenet_accuracy_eval tool.

    Args:
        image: A numpy array in shape of (H, W, 3)
        imsize: height/width size of NN input tensor
        output_type: np.uint8/np.int8/np.float32, output type of NN

    Returns:
        A numpy array (reshaped to (1, imsize, imsize, 3)) after preprocessing
    """
    image = crop_and_resize(image, imsize)

    if output_type == np.uint8:
        mean = 0
        scale = 1
    elif output_type == np.int8:
        mean = 128
        scale = 1
    elif output_type == np.float32:
        mean = 127.5
        scale = 1 / mean
    else:
        assert False

    image = ((image.astype(np.float32) - mean) * scale).astype(output_type)

    return np.reshape(image, (1, imsize, imsize, 3))
