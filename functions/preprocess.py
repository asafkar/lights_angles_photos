
import cv2
import numpy as np
import glob
from functions import cfg
from collections import defaultdict
import os

def create_mask_from_optical_flow(flow, img_size, polar_cords=False):
    mask = np.zeros(img_size, dtype=np.uint8)
    if polar_cords:
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        im_mask = (rgb > cfg.optical_flow_mask_thr).astype(np.uint8) * 255
        im_mask = (cv2.cvtColor(im_mask, cv2.COLOR_RGB2GRAY))

        # Opens a new window and displays the output frame
        cv2.imshow("dense optical flow", im_mask)
        cv2.imshow("dense optical flow", rgb)
        cv2.waitKey(15000)
        cv2.destroyAllWindows()

        return im_mask
    else:
        abs_flow = np.power(flow[..., 0], 2) + np.power(flow[..., 1], 2)
        norm_flow = cv2.normalize(abs_flow, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("dense optical flow", norm_flow)
        # cv2.waitKey(1500)
        # cv2.destroyAllWindows()
        return norm_flow


def img_to_arr_angle_led():
    img_dict = defaultdict()

    for file in glob.glob(cfg.path + "*.jpg"):
        fn = file.split("\\")[-1].split(".jpg")[0]  # pos_0_led_0_050
        pos = "" + (fn.split("pos_")[1].split("_")[0])
        if cfg.const_light:
            led = "const"
        else:
            led = fn.split("led_")[1]
        img = cv2.imread(file)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img = img.astype(int)  # needed for manipulations, can't use default uint8
        if img_dict.get(pos) is None:
            img_dict[pos] = [np.array(img)]
        else:
            img_dict[pos].append(np.array(img))
    return img_dict


# vectorized version of dict/np
def img_led_angl_tensor(path_to_load=cfg.path, const_light=cfg.const_light, grayscale=False, blur=False):
    img_dict = defaultdict()  # this will save file names in-order

    for file in glob.glob(path_to_load + "*.jpg"):
        fn = file.split("\\")[-1].split(".jpg")[0]  # pos_0_led_0_050
        pos = "" + (fn.split("pos_")[1].split("_")[0])
        pos = pos.zfill(3)  # zero pad to 3
        if const_light:
            led = "const"
        else:
            led = fn.split("led_")[1]
        # img = cv2.imread(file)
        # if grayscale:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = img.astype(int)  # needed for manipulations, can't use default uint8
        if img_dict.get(led) is None:
            img_dict[led] = defaultdict()
        img_dict[led][pos] = file

    arbitrary_file = next(iter(next(iter(img_dict.values())).values()))
    img = cv2.imread(arbitrary_file)
    w, h, _ = img.shape
    if cfg.downscale:
        w, h = w//4, h//4
    angles = len(next(iter(img_dict.values())))
    leds = len(img_dict)
    color = 3 if not grayscale else 1
    res = np.empty([leds, angles, w, h, color], dtype=np.uint8).squeeze()  # squeeze if for grayscale
    led, angle = 0, 0

    for key_led, img_l in img_dict.items():  # go over leds
        for key_pos, img_pos in sorted(img_l.items()):  # go over sorted angles
            img_ = cv2.imread(img_pos)
            if grayscale:
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            if cfg.downscale:
                img_ = cv2.resize(img_, (h, w))
            if blur:
                img_ = cv2.GaussianBlur(img_, (7, 7), 0)
            res[led, angle] = img_
            angle += 1
        angle = 0
        led += 1
    return res


def img_to_arr_led_angle(path_to_load=cfg.path, const_light=cfg.const_light, grayscale=False):
    img_dict = defaultdict()

    for file in glob.glob(path_to_load + "*.jpg"):
        fn = file.split("\\")[-1].split(".jpg")[0]  # pos_0_led_0_050
        pos = "" + (fn.split("pos_")[1].split("_")[0])
        pos = pos.zfill(3)  # zero pad to 3
        if const_light:
            led = "const"
        else:
            led = fn.split("led_")[1]
        img = cv2.imread(file)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(int)  # needed for manipulations, can't use default uint8
        if img_dict.get(led) is None:
            img_dict[led] = defaultdict()
        img_dict[led][pos] = np.array(img)

    for key_led, img_l in img_dict.items():  # go over leds
        pos_list = []
        for key_pos, img_pos in sorted(img_l.items()):  # go over sorted angles
            pos_list.append(np.asarray(img_pos))
        img_dict[key_led] = pos_list  # list instead of dict
    return img_dict

def dict_to_list(input_dict):
    out_list = []

    for key, img_l in input_dict.items():
        img_from_angle = np.asarray(img_l)
        out_list.append(img_from_angle)
    return np.asarray(out_list)


# displaying all the images in 3 rows
def display_images(img_tensor, shrink=2, save_output=False):
    print("displaying final image")
    all_images = np.hstack(img_tensor)
    if cfg.const_light:
        all_images = np.vstack([all_images[:, 0:int(all_images.shape[1]/2)],
                                all_images[:, int(all_images.shape[1]/2):]])
    else:
        all_images = np.vstack([all_images[:, 0:int(all_images.shape[1]/3)],
                                all_images[:, int(all_images.shape[1]/3):2*int(all_images.shape[1]/3)],
                                all_images[:, 2*int(all_images.shape[1]/3):]])
    output_img = cv2.resize(all_images, (int(all_images.shape[1]/shrink), int(all_images.shape[0]/shrink)))
    if save_output:
        cv2.imwrite(os.getcwd()+"/data/output_mask.jpg", output_img)
    cv2.imshow("all images", output_img)
    # cv2.imshow("all images", all_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









