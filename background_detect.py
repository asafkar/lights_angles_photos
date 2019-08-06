import numpy as np
import cv2

from collections import defaultdict
from functions import preprocess
from functions import cfg

#%%
img_dict = preprocess.img_to_arr_led_angle()
img_dict_arr_angle = preprocess.img_to_arr_angle_led()
filtered_img = []

# go over all leds, and over each angle
bgrd_R = []
bgrd_G = []
bgrd_B = []

for key, img_specific_led in img_dict.items():
    img_from_led = np.asarray(img_specific_led).astype(int)
    mR = np.median(img_from_led[:, :, :, 0], axis=0)
    mG = np.median(img_from_led[:, :, :, 1], axis=0)
    mB = np.median(img_from_led[:, :, :, 2], axis=0)

    # medImage = np.dstack([mR, mG, mB]).astype(np.uint8)
    # cv2.imshow("background", medImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bgrd_R.append(mR)
    bgrd_G.append(mG)
    bgrd_B.append(mB)


bgrd_diff = 15
bgrd = []
for key, img_l in img_dict_arr_angle.items():
    img_from_angle = np.asarray(img_l)
    R_bfit = (np.abs(img_from_angle[:, :, :, 0]) - bgrd_R < bgrd_diff)
    G_bfit = (np.abs(img_from_angle[:, :, :, 1]) - bgrd_G < bgrd_diff)
    B_bfit = (np.abs(img_from_angle[:, :, :, 2]) - bgrd_B < bgrd_diff)

    tmp_RGB_sum = np.sum(R_bfit, axis=0) + np.sum(G_bfit, axis=0) + np.sum(B_bfit, axis=0)
    bgrd.append(tmp_RGB_sum/np.max(tmp_RGB_sum))
    # cv2.imshow("background", bgrd[-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
se = np.ones((3, 3), np.uint8)
idx = 0
for key, img_l in img_dict_arr_angle.items():
    fgrd = (bgrd[idx] < 0.9).astype(np.uint8)
    fgrd = cv2.morphologyEx(fgrd, cv2.MORPH_CLOSE, se)
    fgrd = cv2.morphologyEx(fgrd, cv2.MORPH_OPEN, se).astype(np.uint8)
    cv2.imshow("foreground", fgrd*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_from_angle = np.asarray(img_l)
    filtered_img.append(img_from_angle*fgrd[:, :, np.newaxis].astype('uint8'))
    idx += 1



mean_filtered_img = []
# mean of the pictures, for the sake of the easier viewing...
for img_from_angle in filtered_img:
    np.sort(img_from_angle, axis=0)
    # img_from_angle[-1] = img_from_angle[-3]
    # img_from_angle[-2] = img_from_angle[-3]
    # img_from_angle[0] = img_from_angle[1]
    img_to_store = np.mean(img_from_angle, axis=0)
    mean_filtered_img.append(img_to_store)


preprocess.display_images(np.asarray(mean_filtered_img, dtype='uint8'))



