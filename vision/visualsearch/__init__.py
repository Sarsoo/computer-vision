from typing import List
import random
import numpy as np
from vision.model import Image
import vision.maths.precision_recall as pr
import logging

logger = logging.getLogger(__name__)


class QueryResult:
    def __init__(self,
                 sorted_images: List[Image],
                 query_image: Image,
                 precision_recall: pr.PrecisionRecall):
        self.sorted_images = sorted_images
        self.query_image = query_image
        self.precision_recall = precision_recall


def run_query(images: List[Image], distance_measure=None, query_index=None):
    logger.info(f'running query on {len(images)} images, query index {query_index}')

    if query_index is not None:
        query_image = images[query_index]
    else:
        query_image = random.choice(images)

    if any(i for i in images if i.descriptor is None):
        raise ValueError('descriptors required for all images')

    for image in images:
        if distance_measure is None:
            image.distance = np.linalg.norm(image.descriptor-query_image.descriptor)
        else:
            image.distance = distance_measure(image.descriptor - query_image.descriptor)

    images = [i for i in images if not (i.category == query_image.category and i.name == query_image.name)]

    query_pr = pr.get_pr(images, query=query_image)

    results = QueryResult(sorted_images=images,
                          query_image=query_image,
                          precision_recall=query_pr)
    logger.info(f'query finished AP: {results.precision_recall.ap}')
    return results
