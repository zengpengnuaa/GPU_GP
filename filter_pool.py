import numpy as np

mean_filter = np.float32([0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111])
gauss_filter = np.float32([0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625])
sharpen_filter = np.float32([-1, -1, -1, -1, 9, -1, -1, -1, -1])
prewitt_r_filter = np.float32([-1, 0, -1, -1, 0, -1, -1, 0, -1])
prewitt_ru_filter = np.float32([0, 1, 1, -1, 0, 1, -1, -1, 0])
sobel_u_filter = np.float32([1, 2, 1, 0, 0, 0, -1, -2, -1])
sobel_lu_filter = np.float32([2, 1, 0, 1, 0, -1, 0, -1, -2])
laplace_4_filter = np.float32([0, 1, 0, 1, -4, 1, 0, 1, 0])
laplace_8_filter = np.float32([1, 1, 1, 1, -8, 1, 1, 1, 1])
log_filter = np.float32([0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, -16, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0])