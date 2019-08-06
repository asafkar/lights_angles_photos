import numpy as np
import cv2

from collections import defaultdict
from functions import preprocess
from functions import cfg

from matplotlib import pyplot as plt
#%%
img_dict = preprocess.img_to_arr_angle_led()
img_dict_thr = defaultdict()
all_img = None


img_list_angle = preprocess.dict_to_list(preprocess.img_to_arr_angle_led())
angles, leds, h, w, _ = img_list_angle.shape

# find the histogram for some const led
# bins = numpy.bincount(np.sum(flat,1)/flat.shape[1],minsize=256)
img_const_led = img_list_angle[:, 0, :, :, :]
img_const_led_flat = img_const_led.reshape(36, h*w, 3)  # turn each image to flat vector
# img_hist_const_led = np.bincount(img_const_led_flat[0, :, 0])   # histogram of

# img_hist_const_led = plt.hist2d(img_const_led_flat)
corner_pixel_arr = img_const_led_flat[:, 0, :]
mid_pixel_arr = img_const_led_flat[:, (h+1)*w//2, :]
plt.hist(corner_pixel_arr, 256, [0, 256])
# plt.hist(mid_pixel_arr, 256, [0, 256])
plt.show()
pass


# go over all image angles. For each image batch of a certain angle, sort the pixels by intensity.
# Fill the two brightest values with 3rd brightest. Take the average and return.
for key, img_l in img_dict.items():
    img_from_angle = np.asarray(img_l)
    # if all_img is None:
    #     all_img = img_from_angle
    # else:
    #     all_img = np.vstack((all_img, img_from_angle))
    if cfg.const_light:
        img_to_store = img_from_angle[0]
    else:
        np.sort(img_from_angle, axis=0)
        img_from_angle[-1] = img_from_angle[-3]
        img_from_angle[-2] = img_from_angle[-3]
        img_from_angle[0] = img_from_angle[1]
        img_to_store = np.mean(img_from_angle, axis=0)
    img_dict_thr[key] = np.array(img_to_store, dtype=np.uint8)



img_thr = [img for key, img in img_dict_thr.items()]
img_thr = np.asarray(img_thr)
# preprocess.display_images(img_thr)


open_cv_bgs = True
if open_cv_bgs:
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (7, 7))
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    for img in img_thr:  # all_img:
        img_background = backgroundSubtractor.apply(img)
        img_background = cv2.morphologyEx(img_background, cv2.MORPH_CROSS, kernel)

else:
    # get image background
    img_thr = np.asarray(img_thr, dtype=int)  # int needed for median
    img_background = np.mean(img_thr, axis=0).astype(np.uint8)
    img_background = cv2.cvtColor(img_background, cv2.COLOR_BGR2GRAY)

cv2.imshow("background", img_background)
cv2.waitKey(0)
cv2.destroyAllWindows()


""" subtract background from images """
# create the mask
mask = np.zeros(shape=img_background.shape).astype('uint8')
mask[img_background > 0] = 1

img_thr = (img_thr*mask[:, :, np.newaxis]).astype('uint8')
# preprocess.display_images(img_thr)





