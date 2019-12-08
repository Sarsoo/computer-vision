from vision.model import Image
from typing import List
import numpy as np
from scipy.signal import convolve2d
import math as m
import cv2


class Edge:
    def __init__(self, magnitude: np.array, angle: np.array):
        self.magnitude = magnitude
        self.angle = angle


def get_edge_angle_hist(edge: Edge, bins: int, threshold: float):

    angle_vals = []
    for i in range(edge.magnitude.shape[0]):
        for j in range(edge.magnitude.shape[1]):
            if edge.magnitude[i, j] > threshold:

                bin_val = m.floor((edge.angle[i, j] / (2*np.pi)) * bins)
                angle_vals.append(bin_val)

    if len(angle_vals) > 0:
        return np.histogram(angle_vals, bins=bins, density=True)[0]
    else:
        return np.zeros(bins)


def get_edge_info(pixels: np.array = None,
                  image: Image = None,
                  images: List[Image] = None,
                  blur: bool = True):

    if pixels is None and image is None and images is None:
        raise KeyError('no image provided')

    def extract(i):
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4

        grey = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

        if blur is True:
            grey = convolve2d(grey, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9)

        dx = convolve2d(grey, kx)
        dy = convolve2d(grey, ky)

        mag = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        angle = (angle + 2 * np.pi) % (2 * np.pi)
        return Edge(mag, angle)

    if images is not None:
        for image in images:
            image.descriptor = extract(image.pixels)
        return
    elif image is not None:
        return extract(image.pixels)
    else:
        return extract(pixels)
