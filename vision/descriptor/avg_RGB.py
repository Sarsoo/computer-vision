from vision.model import Image
from typing import List


def extract_average_rgb(images: List[Image]):
    for image in images:
        image.descriptor = image.mean(axis=(0, 1))
