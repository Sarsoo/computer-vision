from vision.model import Image
from typing import List
import numpy as np


def get_category_histogram(images: List[Image], bins: int):
    return np.histogram([i.category for i in images], bins=bins)
