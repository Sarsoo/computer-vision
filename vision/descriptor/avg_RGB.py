from vision.model import Image
from typing import List
import numpy as np
import logging
logger = logging.getLogger(__name__)


def extract_average_rgb(pixels: np.array = None,
                        image: Image = None,
                        images: List[Image] = None):

    if pixels is None and image is None and images is None:
        raise KeyError('no image provided')

    def extract(i):
        return i.mean(axis=(0, 1))

    if images is not None:
        length = len(images)
        for index, image in enumerate(images):
            logger.debug(f'generating {index} of {length}')
            image.descriptor = extract(image)
        return
    elif image is not None:
        image.descriptor = extract(image)
    else:
        return extract(pixels)
