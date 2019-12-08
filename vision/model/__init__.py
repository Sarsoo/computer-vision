import numpy as np


class Image:
    def __init__(self,
                 pixels: np.array,
                 category=None,
                 name=None,
                 descriptor=None):
        self.pixels = pixels
        self.category = category
        self.name = name
        self.descriptor = descriptor

        self.distance = 0

    def clear(self):
        self.descriptor = None
        self.distance = 0

    @property
    def shape(self):
        return self.pixels.shape

    @property
    def height(self):
        return self.pixels.shape[0]

    @property
    def width(self):
        return self.pixels.shape[1]

    @property
    def T(self):
        return self.pixels.T

    @property
    def flat(self):
        return self.pixels.flat

    @property
    def max(self):
        return self.pixels.max()

    @property
    def min(self):
        return self.pixels.min()

    def round(self, decimals):
        return self.pixels.round(decimals)

    def sum(self, axis=None):
        if axis is not None:
            return self.pixels.sum(axis=axis)
        else:
            return self.pixels.sum()

    def mean(self, axis=None):
        if axis is not None:
            return self.pixels.mean(axis=axis)
        else:
            return self.pixels.mean()

    def __add__(self, other):
        return self.pixels + other

    def __sub__(self, other):
        return self.pixels - other

    def __mul__(self, other):
        return self.pixels * other

    def __truediv__(self, other):
        return self.pixels / other

    def __floordiv__(self, other):
        return self.pixels // other

    def __mod__(self, other):
        return self.pixels % other

    def __pow__(self, other):
        return pow(self.pixels, other)

    def __eq__(self, other):
        return isinstance(other, Image) and self.pixels == other.pixels

    def __repr__(self):
        return f'Image: {self.shape} ({self.descriptor})'
