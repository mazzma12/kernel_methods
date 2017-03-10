import numpy as np
import pandas as pd
from numpy import arctan2

import math
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv


def rgb2gray(rgb, reshape = True):
    # if reshape is true, array is reshape in 32.32 pixels
    n = rgb.shape[0]
    r = rgb[:,:1024]
    g = rgb[:,1024:2048]
    b = rgb[:,2048:]
    gray = 0.299*r + 0.587*g + 0.114*b
    # from matlab 0.2989 * R + 0.5870 * G + 0.1140 * B
    if reshape is True:
        gray = gray.reshape(n, 32,32)
    return gray
    
scharr_x = np.array([-3, 0, 3, -10, 0, 10, -3, 0, 3])
scharr_y = np.array([-3, -10, -3, 0, 0, 0, 3, 10, 3])

def rgb2hsv2(X):
    n = X.shape[0]
    p = X.shape[1]
    X_res = np.zeros_like(X)
    for kk in range(n):
        X_res[kk] = hsv_to_rgb(X[kk].reshape(p/3, 3)).reshape(-1,)
    
    return X_res

"""
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
"""
"""def rgb2hsv(X):
    # X is just an image, not an array
    n = X.shape[0]
    r = X[:n]
    g = X[n/3: n/2]
    b = X[n/2:]
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return np.r_[h, s, v]
"""
def convolution(image_patch):
    Gx = np.dot(scharr_x, image_patch)
    Gy = np.dot(scharr_y, image_patch)
    # return np.sqrt(Gx ** 2 + Gy ** 2)
    return Gx, Gy

def intensity_and_orientation(gx, gy):
    intensity = np.sqrt(gx**2 + gy**2)
    orientation = (arctan2(gy, gx) * 180 / np.pi) % 180
            
    return intensity, orientation

def scharr_gradient(image):
    gx = np.zeros(image.shape)
    gx[:, 1:-1] = -image[:, :-2] + image[:, 2:]
    gx[:, 0] = -image[:, 0] + image[:, 1]
    gx[:, -1] = -image[:, -2] + image[:, -1]
    
    gy = np.zeros(image.shape)
    gy[1:-1, :] = image[:-2, :] - image[2:, :]
    gy[0, :] = image[0, :] - image[1, :]
    gy[-1, :] = image[-2, :] - image[-1, :]
    return intensity_and_orientation(gx, gy)

def scharr_gradients(X, gray = True):
    if gray:
        X_gray = rgb2gray(X, reshape = False)
        return np.apply_along_axis(scharr_gradient, axis = 0, arr = X_gray)
    else:
        X_max_gradients = 0
        colors = ['red', 'green', 'blue']
        image_len = 1024
        for color, color_number in zip(colors, range(3)):
            print('Processing color:', color)
            X_color = X[:, image_len * color_number : image_len * (color_number + 1)]
            X_color_gradients = np.apply_along_axis(scharr_gradient, axis = 1, arr = X_color)
            X_max_gradients = np.maximum(X_max_gradients, X_color_gradients)
        return X_max_gradients
        
def subimage_histogram(intensities, orientations, nbins, b_step):
    left_bins  = (orientations % 180) // b_step
    left_bins  = left_bins % nbins
    right_bins = (left_bins + 1) % nbins
    to_left_bin  = (orientations / b_step - left_bins) * intensities
    to_right_bin = (left_bins + 1 - orientations / b_step) * intensities
    histogram = np.zeros(nbins)
    for i in range(nbins):
        histogram[i] = np.sum((left_bins == i) * to_left_bin + (right_bins == i) * to_right_bin)
    return histogram
        
def image_histogram(intensities, orientations, cell_sz, cells, step, nbins, b_step, image_sz):
    histogram = np.zeros(cells ** 2 * nbins)
    for i in range(0, image_sz, step):
        for j in range(0, image_sz, step):
            sub_intensities  = intensities[i:i + cell_sz, j:j + cell_sz]
            sub_orientations = orientations[i:i + cell_sz, j:j + cell_sz]
            position = (i / step * cells + j / step) * nbins
            histogram[position:position + nbins] = subimage_histogram(sub_intensities, sub_orientations, nbins, b_step)
    return histogram

def histogram_of_gradients(X, nbins, cell_sz, step, colored = True):
    if colored:
        X_gray = rgb2gray(X)
    else:
        X_gray = X
    intensities  = np.zeros(X_gray.shape)
    orientations = np.zeros(X_gray.shape)
    for i, image in enumerate(X_gray):
        intensities[i], orientations[i] = scharr_gradient(image)
    # X_gradients = np.apply_along_axis(lambda image: scharr_gradient(image, only_intensity = False), axis = 1, arr = X_gray)
    b_step = 180/nbins
    image_sz = len(X_gray[0]) #images are supposed to be square
    cells = image_sz // step
    HOG = np.zeros((len(X), cells ** 2 * nbins))
    for i, (intensity, orientation) in enumerate(zip(intensities, orientations)):
        HOG[i] = image_histogram(intensity, orientation, cell_sz, cells, step, nbins, b_step, image_sz)
    return HOG
            
    
def plot(X, lim=4):
    n = X.shape[0]
    for kk in range(n):
        if kk>lim:
            break
        plt.imshow(X[kk], cmap='gray')
        plt.show()

def shift(X, direction):
    n, p = X.shape
    X_r = X[:, :1024].reshape(n, 32, 32)
    X_g = X[:, 1024:2048].reshape(n, 32, 32)
    X_b = X[:, 2048:].reshape(n, 32, 32)
    shifted = np.zeros_like(X)  # Will contain the shifted RGB
    colors = [X_r, X_g, X_b]
    for kk, col in enumerate(colors):
        if (direction == 'right'):
            temp = col[:, :, 0]
            shifted_col = np.roll(col, axis=2, shift=1)
            shifted_col[:, :, 0] = temp
        elif (direction == 'left'):
            temp = col[:, :, -1]
            shifted_col = np.roll(col, axis=2, shift=-1)
            shifted_col[:, :, -1] = temp
        elif (direction == 'up'):
            temp = col[:, -1, :]
            shifted_col = np.roll(col, axis=1, shift=-1)
            shifted_col[:, -1, :] = temp
        elif (direction == 'down'):
            temp = col[:, 0, :]
            shifted_col = np.roll(col, axis=1, shift=1)
            shifted_col[:, 0, :] = temp
        else:
            print('Direction \'%s\'' % direction, 'is not supported')
            return None

        shifted_col = shifted_col.reshape(n, -1)
        shifted[:, 1024 * kk:1024 * (kk + 1)] = shifted_col
    return shifted


def flip_lr(X):
    # Augmentation by flipping (transpose)
    n, p = X.shape
    X_res = np.zeros((n, p))  # FLipped images are assigned here
    X_r = X[:, :1024]
    X_g = X[:, 1024:2048]
    X_b = X[:, 2048:]

    for kk in range(n):
        x_r = np.fliplr(X_r[kk].reshape(32, 32)).ravel()
        x_g = np.fliplr(X_b[kk].reshape(32, 32)).ravel()
        x_b = np.fliplr(X_g[kk].reshape(32, 32)).ravel()
        new_sample = np.r_[x_r, x_g, x_b].reshape(1, -1)
        X_res[kk] = new_sample

    return X_res


def flip_ud(X):
    # Augmentation by flipping (transpose)
    n, p = X.shape
    X_res = np.zeros((n, p))  # FLipped images are assigned here
    X_r = X[:, :1024]
    X_g = X[:, 1024:2048]
    X_b = X[:, 2048:]

    for kk in range(n):
        x_r = np.flipud(X_r[kk].reshape(32, 32)).ravel()
        x_g = np.flipud(X_b[kk].reshape(32, 32)).ravel()
        x_b = np.flipud(X_g[kk].reshape(32, 32)).ravel()
        new_sample = np.r_[x_r, x_g, x_b].reshape(1, -1)
        X_res[kk] = new_sample

    return X_res


def differentiate_right(X):
    n, p = X.shape

    X_r = X[:, :1024].reshape(n, 32, 32)
    X_g = X[:, 1024:2048].reshape(n, 32, 32)
    X_b = X[:, 2048:].reshape(n, 32, 32)
    diff = np.zeros_like(X)  # Will contain the differentiate RGB
    colors = [X_r, X_g, X_b]
    for kk, col in enumerate(colors):
        diff_col = np.zeros_like(col)
        temp = col[:, :, 0]
        diff_col[:, :, 1:] = np.diff(col, axis=2)  # np.roll(col, axis=2, shift=1)
        diff_col[:, :, 0] = temp
        diff_col = diff_col.reshape(n, -1)
        diff[:, 1024 * kk:1024 * (kk + 1)] = diff_col

    return diff


def differentiate_down(X):
    n, p = X.shape

    X_r = X[:, :1024].reshape(n, 32, 32)
    X_g = X[:, 1024:2048].reshape(n, 32, 32)
    X_b = X[:, 2048:].reshape(n, 32, 32)
    diff = np.zeros_like(X)  # Will contain the differentiate RGB
    colors = [X_r, X_g, X_b]
    for kk, col in enumerate(colors):
        diff_col = np.zeros_like(col)
        temp = col[:, 0, ]
        diff_col[:, 1:, :] = np.diff(col, axis=1)  # np.roll(col, axis=2, shift=1)
        diff_col[:, 0, :] = temp
        diff_col = diff_col.reshape(n, -1)
        diff[:, 1024 * kk:1024 * (kk + 1)] = diff_col

    return diff


def differentiate_left(X):
    X = flip_lr(X)
    X = differentiate_right(X)
    X = flip_lr(X)
    return X


def differentiate_up(X):
    X = flip_ud(X)
    X = differentiate_down(X)
    X = flip_ud(X)
    return X
