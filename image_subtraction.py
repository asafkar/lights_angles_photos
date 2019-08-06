from functions import preprocess
import cv2
import numpy as np
import os
import multiprocessing as mp


def img_bgrd_func(apply_grabcut_, subt_image, subt_image_before, subt_image_after, final_img_):
	# noise removal
	kernel = np.ones((3, 3), np.uint8)
	for idx, img_angle in enumerate(img_tensor_avg):
		if apply_grabcut_:
			# apply grabcut on sure foreground
			mask_ = np.median([subt_image // 255, subt_image_before // 255,
					subt_image_after // 255], axis=0)
			mask_ = cv2.morphologyEx(mask_, cv2.MORPH_OPEN, kernel, iterations=2).astype('uint8')
			img = final_img_.astype('uint8')
			bgdModel = np.zeros((1, 65), np.float64)  # used internally by model
			fgdModel = np.zeros((1, 65), np.float64)  # used internally by model
			mask_gb, _, _ = cv2.grabCut(img, mask_, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
			final_mask = np.where((mask_gb == 2) | (mask_gb == 0), 0, 1).astype('uint8')  # convert fg/bg/prlly_bcgr/frgd to 0/1
			final_mask = final_mask*255
		else:
			final_mask = cv2.morphologyEx(img_angle, cv2.MORPH_OPEN, kernel, iterations=3)
		return final_mask


use_max = True  # indicates whether to use max / median when giving images to grabcut / final result
img_minus_background = True
const_light = False
apply_grabcut = True
display_mask = False  # setting this to true will display the mask, else display the image after masking
threshold_early = True  # this means - do the threshold operation first, and then sum. Else - opposite
threshold = 15 if threshold_early else 80
grayscale = False


path = os.getcwd()+"/data/multilight_24_7/giraffe/"
bgrd_path = os.getcwd()+"/data/multilight_24_7/background/"

if img_minus_background:
	img_tensor = preprocess.img_led_angl_tensor(path, const_light=False, grayscale=grayscale).squeeze()
	pure_bgrd_tensor = preprocess.img_led_angl_tensor(bgrd_path, const_light=False, grayscale=grayscale).squeeze()
	mask_tensor = np.abs(img_tensor - pure_bgrd_tensor)  # L1
	# mask_tensor = np.abs(img_tensor*img_tensor - pure_bgrd_tensor*pure_bgrd_tensor)  # L2

	if grayscale:
		mask = (mask_tensor > threshold)
	else:
		if threshold_early:
			mask = (mask_tensor[:, :, :, :, 0] > threshold) | (mask_tensor[:, :, :, :, 1] > threshold) \
					| (mask_tensor[:, :, :, :, 2] > threshold)
			img_tensor_avg = np.median(mask, axis=0)  # try precentile?
		else:
			mask = np.mean(mask_tensor, axis=0)  # sum by led axis   L1
			# mask = np.sqrt(np.sum(mask_tensor, axis=0))  # L2
			img_tensor_avg = (mask[:, :, :, 0] > threshold) | (mask[:, :, :, 1] > threshold) | (mask[:, :, :, 2] > threshold)
else:
	img_tensor = preprocess.img_led_angl_tensor(path, const_light=const_light, grayscale=grayscale, blur=False).squeeze()
	mask_tensor_plus = np.abs(img_tensor - np.roll(img_tensor, axis=1, shift=1)).astype(int)
	mask_tensor_minus = np.abs(img_tensor - np.roll(img_tensor, axis=1, shift=-1)).astype(int)
	mask_tensor = (mask_tensor_plus + mask_tensor_minus) / 2  # average angle backwards and angle forward

leds, angles, h, w, _ = img_tensor.shape

if img_minus_background:
	img_tensor_avg = img_tensor_avg.astype('uint8')*255
	final_img_median = np.median(img_tensor, axis=0)
	final_img_max = np.max(img_tensor, axis=0)

	if use_max:
		final_img = np.max(img_tensor, axis=0)
	else:
		final_img = np.median(img_tensor, axis=0)

	pool = mp.Pool(mp.cpu_count())
	results = [pool.apply_async(img_bgrd_func, args=(apply_grabcut, img_tensor_avg[idx], img_tensor_avg[idx-1],
			img_tensor_avg[(idx + 1) % angles], final_img[idx])) for idx, _ in enumerate(img_tensor_avg)]
	img_tensor_avg = np.asarray([p.get() for p in results])
	pool.close()

	if display_mask:
		final_img = img_tensor_avg
	else:
		img_tensor_avg = img_tensor_avg / 255
		final_img = final_img*np.expand_dims(img_tensor_avg, axis=-1)

	preprocess.display_images(final_img, 1, save_output=True)

else:
	mask_tensor = np.asarray(mask_tensor, dtype=np.uint8)

	# try adding their normals L1/L2 and threshold that..
	mask = (mask_tensor[:, :, :, :, 0] > threshold) | (mask_tensor[:, :, :, :, 1] > threshold) | (mask_tensor[:, :, :, :, 2] > threshold)
	img_tensor = img_tensor*np.tile(mask[:, :, :, :, np.newaxis], 3)

	img_tensor_avg = np.median(img_tensor, axis=0)
	img_tensor_avg = img_tensor_avg.astype('uint8')
	preprocess.display_images(img_tensor_avg, 1, save_output=True)

	# to see each led's mask -
	see_each_led_pov = False
	if see_each_led_pov:
		for led_img in img_tensor:
			preprocess.display_images((led_img.astype('uint8')), 1)


