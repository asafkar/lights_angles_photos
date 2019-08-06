from collections import defaultdict
from functions import preprocess
from functions import cfg
import torch
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import kstest

img_tensor = preprocess.dict_to_list(preprocess.img_to_arr_led_angle())
leds, angles, h, w, _ = img_tensor.shape

# find the histogram for some const led
img_const_led = img_tensor[0, :, :, :, :]  # certain fixed light...
img_const_led_flat = img_const_led.reshape(36, h*w, 3)  # turn each image to flat vector

# corner_pixel_arr = img_const_led_flat[:, 0, :]
# mid_pixel_arr = img_const_led_flat[:, (h+1)*w//2, :]
# plt.hist(corner_pixel_arr, 256, [0, 256])
# plt.hist(mid_pixel_arr, 256, [0, 256])
# plt.show()

histogram_understanding = True
if histogram_understanding:
	pixel_filter = np.zeros([h*w])  # 1 = keep pixel
	for pixel_idx in range(img_const_led_flat.shape[1]):
		curr_pixel = img_const_led_flat[:, pixel_idx, :]  # pixel at all angles and rgb
		hist = []
		# bin for each color intensity, for each r/g/b
		hist.append([ii for ii in np.bincount(curr_pixel[:, 0], minlength=256) if ii != 0])
		hist.append([ii for ii in np.bincount(curr_pixel[:, 1], minlength=256) if ii != 0])
		hist.append([ii for ii in np.bincount(curr_pixel[:, 2], minlength=256) if ii != 0])

		if abs(np.mean(hist[0]) - np.median(hist[0])) < 4 and \
				abs(np.mean(hist[1]) - np.median(hist[1])) < 4 and \
				abs(np.mean(hist[2]) - np.median(hist[2])) < 4:
			threshold = 10
			if len(hist[0]) > threshold and len(hist[1]) > threshold and len(hist[2]) > threshold:
				pixel_filter[pixel_idx] = 1  # white pixel


	pixel_filter = pixel_filter.reshape(h, w)
	# cv2.imshow("all images", pixel_filter*255)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


## tryout - for each depth of pixel (36 rotations), see how much change (max-min)
# max_img = np.max(img_const_led_flat, axis=0)
# min_img = np.min(img_const_led_flat, axis=0)
#
# diff_minmax_vals = np.asarray(abs(max_img - min_img) > 90, dtype=np.uint8)*255
#
# diff_minmax_vals = diff_minmax_vals.reshape(h, w, 3)
# cv2.imshow("background", diff_minmax_vals)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = np.asarray(img_const_led[5], dtype=np.uint8)
mask = np.asarray(pixel_filter, dtype=np.uint8)
bgdModel = np.zeros((1, 65), np.float64)  # used internally by model
fgdModel = np.zeros((1, 65), np.float64)  # used internally by model
# rect = (w//2, h//2, w//2, h)  # (x,y,w,h) - definite image
cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img*mask2[:, :, np.newaxis]
plt.imshow(img), plt.colorbar(), plt.show()
