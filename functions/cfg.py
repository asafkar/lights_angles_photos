

# this file holds the configurations
const_light = False

# if const_light:
#     path = "C:\\Users\\Asaf Karnieli\\Google Drive\\msc\\thesis\\data\\cam_case_const_light\\"
# else:
#     path = "C:\\Users\\Asaf Karnieli\\Google Drive\\msc\\thesis\\data\\cam_case_6led\\"

if const_light:
    path = "C:\\Users\\Asaf Karnieli\\Google Drive\\msc\\thesis\\data\\box_const_light_36_angle\\"
else:
    path = "C:\\Users\\Asaf Karnieli\\Google Drive\\msc\\thesis\\data\\box_6_led_36_angle\\"


optical_flow_mask_thr = 30
take_both_directions = False  # optical flow - take flow of images (1,2) and of (2,1) and take mean

downscale = False  # downscale images when opening them
