from vision.model import Image
from typing import List
import numpy as np

import vision.descriptor.avg_RGB as rgb
import vision.util.edge as edge
import logging
logger = logging.getLogger(__name__)


def grid_image(height: int, width: int, pixels: np.array):
    shape = pixels.shape

    segments = []
    for i in range(height):
        for j in range(width):

            row_start = round(i * shape[0] / height)
            row_end = round((i+1) * shape[0] / height)

            column_start = round(j * shape[1] / width)
            column_end = round((j + 1) * shape[1] / width)

            segments.append(pixels[row_start:row_end, column_start:column_end, :])

    return segments


def extract_spatial_texture(height: int,
                            width: int,
                            bins: int,
                            threshold: float,
                            pixels: np.array = None,
                            image: Image = None,
                            images: List[Image] = None):

    if pixels is None and image is None and images is None:
        raise KeyError('no image provided')

    def extract(i):
        segments = grid_image(height, width, i)
        descriptor = np.array([])
        for seg in segments:
            img_edge = edge.get_edge_info(pixels=seg)
            hist = edge.get_edge_angle_hist(img_edge, bins=bins, threshold=threshold)
            descriptor = np.append(descriptor, hist[0])
        return descriptor

    if images is not None:
        length = len(images)
        for index, image in enumerate(images):
            logger.debug(f'generating {index} of {length}')
            image.descriptor = extract(image.pixels)
        return
    elif image is not None:
        image.descriptor = extract(image.pixels)
    else:
        return extract(pixels)


def extract_spatial_average_rgb(height: int,
                                width: int,
                                pixels: np.array = None,
                                image: Image = None,
                                images: List[Image] = None):

    if pixels is None and image is None and images is None:
        raise KeyError('no image provided')

    def extract(i):
        segments = grid_image(height, width, pixels)
        descriptor = np.array([])
        for seg in segments:
            descriptor = np.append(descriptor, rgb.extract_average_rgb(pixels=seg))
        return descriptor

    if images is not None:
        length = len(images)
        for index, image in enumerate(images):
            logger.debug(f'generating {index} of {length}')
            image.descriptor = extract(image.pixels)
        return
    elif image is not None:
        image.descriptor = extract(image.pixels)
    else:
        return extract(pixels)
