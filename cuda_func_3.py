import math
from numba import cuda
import numpy as np
import filter_pool


@cuda.jit
def cudafunc(imgs, ret_vectors):
    x = cuda.grid(1)

    def neg(ter):
        return -ter

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

    def mean(img):
        sum = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                sum += img[i][j]
        arg_mean = sum / (len(img) * len(img[0]))
        return arg_mean

    def std(img):
        fangcha_sum = 0
        sum = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                sum += img[i][j]
        arg_mean = sum / (len(img) * len(img[0]))
        for i in range(len(img)):
            for j in range(len(img[0])):
                fangcha_sum += (img[i][j] - arg_mean) * (img[i][j] - arg_mean)
        fangcha = fangcha_sum / (len(img) * len(img[0]))
        arg_std = math.sqrt(fangcha)
        return arg_std

    def relu(left):
        for i in range(len(left)):
            for j in range(len(left[0])):
                left[i][j] = (abs(left[i][j]) + left[i][j]) / 2
        return left

    def conv(pixelset, filter):
        length = int(len(filter))
        size = int(math.sqrt(length))
        img_max = cuda.local.array((38, 38), np.float32)
        img_padding = img_max[0:len(pixelset) + size - 1, 0:len(pixelset) + size - 1]
        ret_img_max = cuda.local.array((32, 32), np.float32)
        ret_img = ret_img_max[0:len(pixelset), 0:len(pixelset)]
        for i in range(len(img_padding)):
            for j in range(len(img_padding[0])):
                img_padding[i, j] = 0
        padding = int((size - 1) / 2)
        for k in range(padding, len(img_padding) - padding):
            for l in range(padding, len(img_padding) - padding):
                img_padding[k, l] = pixelset[k - padding, l - padding]
        for m in range(padding, len(img_padding) - padding):
            for n in range(padding, len(img_padding) - padding):
                window_data = img_padding[m - padding:m + padding + 1, n - padding:n + padding + 1]
                sum = 0
                for p in range(len(window_data)):
                    for q in range(len(window_data)):
                        sum += window_data[p, q] * filter[p * size + q]
                ret_img[m - padding, n - padding] = sum
        return ret_img

    def gen_filter(filter_size, filter_type):
        filter_max = cuda.local.array(25, np.float32)
        filter = filter_max[:filter_size * filter_size]
        if filter_size == 5:
            for i in range(len(filter)):
                filter[i] = filter_pool.log_filter[i]
        else:
            if filter_type == 'mean':
                for i in range(len(filter)):
                    filter[i] = filter_pool.mean_filter[i]
            elif filter_type == 'gauss':
                for i in range(len(filter)):
                    filter[i] = filter_pool.gauss_filter[i]
            elif filter_type == 'sharpen':
                for i in range(len(filter)):
                    filter[i] = filter_pool.sharpen_filter[i]
            elif filter_type == 'prewitt_r':
                for i in range(len(filter)):
                    filter[i] = filter_pool.prewitt_r_filter[i]
            elif filter_type == 'prewitt_ru':
                for i in range(len(filter)):
                    filter[i] = filter_pool.prewitt_ru_filter[i]
            elif filter_type == 'sobel_u':
                for i in range(len(filter)):
                    filter[i] = filter_pool.sobel_u_filter[i]
            elif filter_type == 'sobel_lu':
                for i in range(len(filter)):
                    filter[i] = filter_pool.sobel_lu_filter[i]
            elif filter_type == 'laplace_4':
                for i in range(len(filter)):
                    filter[i] = filter_pool.laplace_4_filter[i]
            elif filter_type == 'laplace_8':
                for i in range(len(filter)):
                    filter[i] == filter_pool.laplace_8_filter[i]
        return filter

    def maxP(left, kelh, kelw):
        ret_img_max = cuda.local.array((32, 32), np.float32)
        height = len(left) - kelh + 1
        width = len(left[0]) - kelw + 1
        ret_img = ret_img_max[:height, :width]
        for i in range(height):
            for j in range(width):
                max = -10000
                for m in range(i, i + kelh):
                    for n in range(j, j + kelw):
                        if left[m, n] > max:
                            max = left[m, n]
                ret_img[i, j] = max
        return ret_img

    def concat(l0, l1, l2, l3, l4, l5, l6, l7, l8, l9):
        feature_vec = cuda.local.array(10, np.float32)
        feature_vec[0] = l0
        feature_vec[1] = l1
        feature_vec[2] = l2
        feature_vec[3] = l3
        feature_vec[4] = l4
        feature_vec[5] = l5
        feature_vec[6] = l6
        feature_vec[7] = l7
        feature_vec[8] = l8
        feature_vec[9] = l9
        return feature_vec

    innerfunc = lambda ARG0: concat(mean(regions(ARG0, 6, 10, 13)), mean(maxP(conv(conv(conv(conv(ARG0, gen_filter(5, 'prewitt_r')), gen_filter(5, 'laplace_8')), gen_filter(3, 'prewitt_r')), gen_filter(3, 'sharpen')), 4, 4)), sub(ARG0, 9, 3, 31, 11), mixsub(std(maxP(conv(ARG0, gen_filter(5, 'laplace_8')), 4, 4)), 0.848, mean(maxP(ARG0, 4, 4)), 0.103), mean(conv(relu(conv(relu(maxP(ARG0, 2, 2)), gen_filter(3, 'sharpen'))), gen_filter(3, 'laplace_4'))), std(ARG0), std(maxP(conv(ARG0, gen_filter(3, 'laplace_4')), 4, 4)), mean(ARG0), mean(conv(ARG0, gen_filter(3, 'sharpen'))), mean(maxP(ARG0, 4, 2)))
    feature_vec = innerfunc(imgs[x])
    for i in range(10):
        ret_vectors[x][i] = feature_vec[i]