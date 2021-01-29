from dataclasses import dataclass
from typing import List
from vision.model import Image


@dataclass
class PrecisionRecall:
    precision: float
    recall: float
    ap: float


def get_precision(images: List[Image], test):
    return len([i for i in images if i.category == test]) / len(images)


def get_recall(images: List[Image], test, category_count):
    return len([i for i in images if i.category == test]) / category_count


def get_pr(images: List[Image], query: Image):
    images = sorted(images, key=lambda x: x.distance)

    query_category_count = len([i for i in images if i.category == query.category])

    p = []
    r = []
    for i in range(len(images)):
        p.append(get_precision(images[:i+1], query.category))
        r.append(get_recall(images[:i+1], query.category, query_category_count))

    precision_list = []
    for index, image in enumerate(images):
        if image.category == query.category:
            precision_list.append(p[index])

    ap = sum(precision_list) / query_category_count

    return PrecisionRecall(precision=p, recall=r, ap=ap)
