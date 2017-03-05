import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

scharr_x = np.array([-3, 0, 3, -10, 0, 10, -3, 0, 3])
scharr_y = np.array([-3, -10, -3, 0, 0, 0, 3, 10, 3])

def convolution(image_patch):
    Gx = np.dot(scharr_x, image_patch)
    Gy = np.dot(scharr_y, image_patch)
    return np.sqrt(Gx ** 2 + Gy ** 2)

def scharr_gradient(image):
    sz = 32 #image size is 32x32
    deltas = [-1, 0, 1]
    indexes = lambda i, j: [(i + di) * sz + j - dj for di in deltas for dj in deltas]
    return np.array([convolution(image[indexes(i, j)]) for i in range(1, sz - 1) for j in range(1, sz - 1)])

def scharr_gradients(X):
    return X.apply(scharr_gradient, axis = 1)

def rgb2gray(rgb, reshape = True):
    r = rgb[:1024].reshape(-1,1)
    g = rgb[1024:2048].reshape(-1,1)
    b = rgb[2048:].reshape(-1,1)
    colors = np.c_[r, g, b]
    gray = np.dot(colors, [0.299, 0.587, 0.114])
    # from matlab 0.2989 * R + 0.5870 * G + 0.1140 * B
    if reshape is True:
        gray = gray.reshape(32,32)
    return gray
    #return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def plot(X, gray=True, lim=4):
    # lim : set a limit number not to crash
    n, p = X.shape
    for kk in range(n):
        if kk>lim:
            break
        x = X[kk]
        im = rgb2gray(x)
        plt.imshow(im, cmap='gray')
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
