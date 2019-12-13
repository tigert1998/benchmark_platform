import numpy as np
import cv2

CROPPING_FRACTION = 0.875
IMSIZE = 224


def std_preprocess_single_image(image, imsize: int, output_type):
    original_height, original_width, channels = image.shape
    assert channels == 3

    ratio = max(
        imsize / (CROPPING_FRACTION * original_width),
        imsize / (CROPPING_FRACTION * original_height)
    )
    crop_height = round(imsize / ratio)
    crop_width = round(imsize / ratio)
    left = round((original_width - crop_width) / 2.0)
    top = round((original_height - crop_height) / 2.0)
    image = image[top:top+crop_height, left:left + crop_width, :]

    image = cv2.resize(
        image, (imsize, imsize),
        interpolation=cv2.INTER_LINEAR)

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

    return image


def std_preprocess(images, imsize: int, output_type):
    """Standard Preprocess.

    Preprocess an array of RGB images (np.array) according to 
    tensorflow lite imagenet_accuracy_eval tool.

    Args:
        images: An array of numpy arrays in shape of (H, W, C)
        imsize: height/width size of NN input tensor
        output_type: np.uint8/np.int8/np.float32, output type of NN

    Returns:
        An array of numpy arrays after preprocessing
    """

    res = []
    for image in images:
        res.append(std_preprocess_single_image(image, imsize, output_type))
    return res
