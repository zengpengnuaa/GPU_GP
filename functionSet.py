import math
import numpy as np
import skimage
from scipy import ndimage
import skimage.measure
import filter_pool


def num_add(lter, rter):
    return lter + rter


def num_sub(lter, rter):
    return lter - rter


def num_mul(lter, rter):
    return lter * rter


def protected_div(lter, rter):
    return lter / rter if rter != 0 else 0


def if_else(cond, lter, rter):
    return lter if cond < 0 else rter


def mixadd(lter, w1, rter, w2):
    return lter * w1 + rter * w2


def mixsub(lter, w1, rter, w2):
    return lter * w1 - rter * w2


def sin(ter):
    return math.sin(ter)


def cos(ter):
    return math.cos(ter)


def tan(ter):
    return math.tan(ter)


def neg(ter):
    return -ter


def AND(lter, rter):
    return (lter and rter)


def OR(lter, rter):
    return (lter or rter)


def mean(*args):
    arg_arr = np.asarray(args)
    arg_mean = np.mean(arg_arr)
    return arg_mean


def std(*args):
    arg_arr = np.asarray(args)
    arg_std = np.std(arg_arr)
    return arg_std


def random_filters(filter_size):
    filters = []
    for i in range(filter_size * filter_size):
        filters.append(np.random.randint(-5, 5))
    return np.asarray(filters)


def sqrt(left):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.sqrt(left, )
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def conv(pixelset, filter):
    length = len(filter)
    size = int(sqrt(length))
    filters_resize = np.asarray(filter).reshape(size, size)
    img = ndimage.convolve(pixelset, filters_resize)
    return img
    # index = 0
    # templist = []
    # resultlist = []
    # Pixelseq = np.atleast_1d(pixelset)
    # for item in Pixelseq:
    #     templist.append(item)
    #     index += 1
    #     if index == len(filter):
    #         fsum = 0
    #         for i in range(index):
    #             fsum += templist[i]*filter[i]
    #         templist = []
    #         resultlist.append(fsum)
    #         index = 0
    # fsum = 0
    # for i in range(index):
    #     fsum += templist[i] * filter[i]
    # resultlist.append(fsum)


def maxP(left, kel1, kel2):
    try:
        Pixelset = skimage.measure.block_reduce(left, (kel1, kel2), np.max)
    except ValueError:
        Pixelset = left
    return Pixelset
    # kernel_size = kel1*kel2
    # index = 0
    # templist = []
    # maxlist = []
    # left = np.atleast_1d(left)
    # for item in left:
    #     templist.append(item)
    #     index += 1
    #     if index==kernel_size:
    #         templist = np.asarray(templist)
    #         max = np.max(templist)
    #         maxlist.append(max)
    #         templist = []
    #         index = 0
    # if templist:
    #     templist = np.asarray(templist)
    #     max = np.max(templist)
    #     maxlist.append(max)
    # maxlist = np.asarray(maxlist)
    # return maxlist


def relu(left):
    return (abs(left) + left) / 2


def add(img, row1, col1, row2, col2):
    left = img[row1][col1]
    right = img[row2][col2]
    return left + right


def sub(img, row1, col1, row2, col2):
    left = img[row1][col1]
    right = img[row2][col2]
    return left - right


def mul(img, row1, col1, row2, col2):
    left = img[row1][col1]
    right = img[row2][col2]
    return left * right


def div(img, row1, col1, row2, col2):
    left = img[row1][col1]
    right = img[row2][col2]
    if right < 0.00001:
        return left
    return left / right


def regions(left, x, y, windowSize):
    width, height = left.shape
    x_end = min(width, x + windowSize)
    y_end = min(height, y + windowSize)
    slice = left[x:x_end, y:y_end]
    return slice


def regionr(left, x, y, windowSize1, windowSize2):
    width, height = left.shape
    x_end = min(width, x + windowSize1)
    y_end = min(height, y + windowSize2)
    slice = left[x:x_end, y:y_end]
    return slice


def concat(*args):
    feature_arr = np.asarray(args)
    return feature_arr


def gen_filter(filter_size, filter_type):
    if filter_size == 5:
        filter = filter_pool.log_filter
    else:
        if filter_type == 'mean':
            filter = filter_pool.mean_filter
        elif filter_type == 'gauss':
            filter = filter_pool.gauss_filter
        elif filter_type == 'sharpen':
            filter = filter_pool.sharpen_filter
        elif filter_type == 'prewitt_r':
            filter = filter_pool.prewitt_r_filter
        elif filter_type == 'prewitt_ru':
            filter = filter_pool.prewitt_ru_filter
        elif filter_type == 'sobel_u':
            filter = filter_pool.sobel_u_filter
        elif filter_type == 'sobel_lu':
            filter = filter_pool.sobel_lu_filter
        elif filter_type == 'laplace_4':
            filter = filter_pool.laplace_4_filter
        elif filter_type == 'laplace_8':
            filter = filter_pool.laplace_8_filter
    return filter
