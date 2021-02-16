import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import pandas as pd
import darknet
from PIL import Image

import download as dl
import register as rg
import const

R = 608
OVER = 51

csv_path = 'detect_crater.csv'
area_csv_path = 'area.csv'


def parser():
	parser = argparse.ArgumentParser(description="YOLO Object Detection")
	parser.add_argument("lat", type=float)
	parser.add_argument("lon", type=float)
	parser.add_argument("r_lat", type=float)
	parser.add_argument("r_lon", type=float)
	parser.add_argument("--weights", default="./task_crater/backup/yolov4-custom_best.weights",
						help="yolo weights path")
	parser.add_argument("--config_file", default="./task_crater/yolov4-custom.cfg",
						help="path to config file")
	parser.add_argument("--data_file", default="./task_crater/crater.data",
						help="path to data file")
	parser.add_argument("--thresh", type=float, default=.70,
						help="remove detections with lower confidence")
	return parser.parse_args()


def check_arguments_errors(args):
	assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
	if not os.path.exists(args.config_file):
		raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
	if not os.path.exists(args.weights):
		raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
	if not os.path.exists(args.data_file):
		raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))


def check_batch_shape(images, batch_size):
	"""
		Image sizes should be the same width and height
	"""
	shapes = [image.shape for image in images]
	if len(set(shapes)) > 1:
		raise ValueError("Images don't have same shape")
	if len(shapes) > batch_size:
		raise ValueError("Batch size higher than number of images")
	return shapes[0]


def load_images(images_path):
	"""
	If image path is given, return it directly
	For txt file, read it and return each line as image path
	In other case, it's a folder, return a list with names of each
	jpg, jpeg and png file
	"""
	input_path_extension = images_path.split('.')[-1]
	if input_path_extension in ['jpg', 'jpeg', 'png']:
		return [images_path]
	elif input_path_extension == "txt":
		with open(images_path, "r") as f:
			return f.read().splitlines()
	else:
		return glob.glob(
			os.path.join(images_path, "*.jpg")) + \
			glob.glob(os.path.join(images_path, "*.png")) + \
			glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
	width = darknet.network_width(network)
	height = darknet.network_height(network)

	darknet_images = []
	for image in images:
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_resized = cv2.resize(image_rgb, (width, height),
								   interpolation=cv2.INTER_LINEAR)
		custom_image = image_resized.transpose(2, 0, 1)
		darknet_images.append(custom_image)

	batch_array = np.concatenate(darknet_images, axis=0)
	batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
	darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
	return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_resized, darknet_image, network, class_names, class_colors, thresh):
	img = Image.fromarray(image_resized)
	img = img.convert('RGB')
	img = np.array(img, dtype=np.uint8)
	# img = np.transpose(img, (2, 0, 1))

	darknet.copy_image_from_bytes(darknet_image, img.tobytes())
	detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
	darknet.free_image(darknet_image)
	image = darknet.draw_boxes(detections, img, class_colors)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
					thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
	image_height, image_width, _ = check_batch_shape(images, batch_size)
	darknet_images = prepare_batch(images, network)
	batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
													 image_height, thresh, hier_thresh, None, 0, 0)
	batch_predictions = []
	for idx in range(batch_size):
		num = batch_detections[idx].num
		detections = batch_detections[idx].dets
		if nms:
			darknet.do_nms_obj(detections, num, len(class_names), nms)
		predictions = darknet.remove_negatives(detections, class_names, num)
		images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
		batch_predictions.append(predictions)
	darknet.free_batch_detections(batch_detections, batch_size)
	return images, batch_predictions


def convert2relative(image, bbox):
	"""
	YOLO format use relative coordinates for annotation
	"""
	x, y, w, h = bbox
	height, width, _ = image.shape
	return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
	"""
	Files saved with image_name.txt and relative coordinates
	"""
	file_name = name.split(".")[:-1][0] + ".txt"
	with open(file_name, "w") as f:
		for label, confidence, bbox in detections:
			x, y, w, h = convert2relative(image, bbox)
			label = class_names.index(label)
			f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
	args = parser()
	check_arguments_errors(args)
	batch_size = 3
	random.seed(3)  # deterministic bbox colors
	network, class_names, class_colors = darknet.load_network(
		args.config_file,
		args.data_file,
		args.weights,
		batch_size=batch_size
	)
	image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
	images = [cv2.imread(image) for image in image_names]
	images, detections,  = batch_detection(network, images, class_names,
										   class_colors, batch_size=batch_size)
	for name, image in zip(image_names, images):
		cv2.imwrite(name.replace("data/", ""), image)
	print(detections)


def get_latlon(nac, h, w):
	lat = float(nac.pos[8]) - (float(nac.pos[8]) - float(nac.pos[6])) * h / float(nac.data['IMAGE_LINES'])
	lon = float(nac.pos[9]) + (float(nac.pos[3]) - float(nac.pos[9])) * w / float(nac.data['LINE_SAMPLES'])
	return lat, lon


def calc_area(nac):
	area = nac.image_lines*nac.scaled_pixel_height * nac.line_samples*nac.scaled_pixel_width
	return area / 1000 / 1000


def detect_nac(nac, df, model, thresh):
	network, class_names, class_colors = model
	nac_img = nac.image
	height, width = nac_img.shape
	for y in range(0, height, R-OVER):
		for x in range(0, width, R-OVER):
			width = darknet.network_width(network)
			height = darknet.network_height(network)
			darknet_image = darknet.make_image(width, height, 3)

			clp = rg.convert_to_uint8(nac_img[y:y+R, x:x+R])
			image_rgb = cv2.cvtColor(clp, cv2.COLOR_BGR2RGB)
			image_resized = cv2.resize(clp, (width, height),
									   interpolation=cv2.INTER_LINEAR)

			image, detections = image_detection(
				image_resized, darknet_image, network, class_names, class_colors, thresh
			)
			if len(detections) > 0:
				lat, lon = get_latlon(nac, y, x)
				for detection in detections:
					img_name = '{}_{}_{}'.format(nac.file_name, y, x)
					cv2.imwrite('/hdd_mount/DETECT_CRATER/{}.png'.format(img_name), image)
					record = pd.Series([img_name, lat, lon, nac.file_name, y, x, 
					','.join(map(str, detection[2])),
				 	str(detection[1]), nac.resolution], index=df.columns)
					df = df.append(record, ignore_index=True)
	return df


def main():
	args = parser()
	check_arguments_errors(args)

	random.seed(3)  # deterministic bbox colors
	model = darknet.load_network(
		args.config_file,
		args.data_file,
		args.weights,
		batch_size=1
	)

	lat = args.lat
	lon = args.lon
	r_lat = args.r_lat
	r_lon = args.r_lon

	print([lat, lon], r_lat, r_lon)
	data = dl.get_data_from_point([lat, lon])
	data = data[data.INCIDENCE_ANGLE < 50]

	print(len(data))
	dl.download_nac_all(data)

	cols = ['IMG_NAME', 'LAT', 'LON', 'NAC', 'W', 'H', 'BBOX', 'SCORE', 'RESOLUTION']
	df = pd.DataFrame(index=[], columns=cols)
	area_df = pd.DataFrame(index=[], columns=['AREA'])

	for i in range(len(data)):
		nac = rg.NacImage(data.iloc[i])
		df = detect_nac(nac, df, model, args.thresh)

		record = pd.Series([calc_area(nac)], index=area_df.columns)
		area_df = area_df.append(record, ignore_index=True)

	df.to_csv(csv_path)
	area_df.to_csv(area_csv_path)


if __name__ == "__main__":
	# unconmment next line for an example of batch processing
	# batch_detection_example()
	main()
