import glob
import os
import pickle
import logging
from typing import List
import cv2
from vision.model import Image

logger = logging.getLogger(__name__)


def load_path(path: str) -> List[str]:
    if not os.path.exists(path):
        logger.error(f'folder {path} does not exist')
        raise FileNotFoundError('path does not exist')

    files = []
    for extension_set in [glob.glob(f'{path}/*.%s' % ext) for ext in ["jpg", "bmp", "png"]]:
        if len(extension_set) > 0:
            files += extension_set
    return files


def load_set(path: str) -> List[Image]:
    logger.info(f'loading set from {path}')

    files = load_path(path)
    images = [Image(cv2.imread(i))[:, :, ::-1] for i in files]
    return images


def load_msrc(path: str, descriptor_path=None) -> List[Image]:
    logger.info(f'loading msrc from {path}, descriptor path {descriptor_path}')

    files = load_path(path)

    images = []
    for image in files:
        file_name = image.split('/')[-1]
        file_name_split = file_name.split('_')
        category = int(file_name_split[0])
        name = int(file_name_split[1])
        images.append(Image(cv2.imread(image)[:, :, ::-1],
                            category=category,
                            name=name))

    if descriptor_path is not None:
        load_descriptors(descriptor_path, images)

    return images


def save_descriptors(images: List[Image], path: str = 'descriptors/default'):
    logger.info(f'saving {len(images)} descriptors to {path}')

    if not os.path.exists(path):
        os.makedirs(path)

    counter = 0
    for image in images:
        if image.name is not None and image.category is not None:
            name = f'{image.category}_{image.name}'
        else:
            name = f'{counter}'
            counter += 1

        with open(os.path.join(path, name), 'wb') as file:
            pickle.dump(image.descriptor, file)


def load_descriptors(path: str = 'descriptors/default', images: List[Image] = None):
    logger.info(f'loading descriptors from {path}, {len(images)} images')

    if not os.path.exists(path):
        logger.error(f'folder {path} does not exist')
        raise FileNotFoundError('folder does not exist')

    descriptors = []
    for file_name in [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))]:

        with open(os.path.join(path, file_name), 'rb') as file:
            descriptor = pickle.load(file)

        if images is not None:
            desc_cat = int(file_name.split('_')[0])
            desc_name = int(file_name.split('_')[1])

            image = next((i for i in images if i.category == desc_cat and i.name == desc_name), None)
            if image is not None:
                image.descriptor = descriptor
            else:
                logger.error(f'no corresponding image found to hold descriptor {file_name}')
        else:
            descriptors.append(descriptor)
    return descriptors
