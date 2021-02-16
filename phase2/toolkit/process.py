import glob, os
import random
import numpy as np
import cv2

import const

EXT = '.png'


def read_data(file_path, ext):
	if not os.path.exists(file_path + '.txt'):
		return -1
	img = cv2.imread(file_path + ext)
	ret = []
	with open(file_path + '.txt') as f:
		lines = [s.strip() for s in f.readlines()]
	for line in lines:
		line = line.split(' ')
		tmp_ret = {}
		tmp_ret['object_class'] = int(line[0])
		tmp_ret['x-center'] = float(line[1])
		tmp_ret['y-center'] = float(line[2])
		tmp_ret['width'] = float(line[3])
		tmp_ret['height'] = float(line[4])
		ret.append(tmp_ret)
	return img, ret


def ano_data_to_txt(data, path, kind):
	ret = ''
	for d in data:
		ret += '{} {} {} {} {}\n'.format(d['object_class'], d['x-center'], d['y-center'], d['width'], d['height']) 
	with open(path+'-'+kind+'.txt', mode='w') as f:
		f.writelines(ret)


def flip(img, data, path):
	ret = []
	flip_img = cv2.flip(img, 1)
	for d in data:
		d['x-center'] = 1 - d['x-center']
	if random.choice([True, False]):
		cv2.imwrite(path+'-flip'+EXT, flip_img)
		ano_data_to_txt(data, path, 'flip')
		ret.append(path+'-flip'+EXT)

	vflip_img = cv2.flip(flip_img, 0)
	for d in data:
		d['y-center'] = 1 - d['y-center']
	if random.choice([True, False]):
		cv2.imwrite(path+'-vflip'+EXT, vflip_img)
		ano_data_to_txt(data, path, 'vflip')
		ret.append(path+'-vflip'+EXT)

	fvflip_img = cv2.flip(vflip_img, 1)
	for d in data:
		d['x-center'] = 1 - d['x-center']
	if random.choice([True, False]):
		cv2.imwrite(path+'-fvflip'+EXT, fvflip_img)
		ano_data_to_txt(data, path, 'fvflip')
		ret.append(path+'-fvflip'+EXT)
	for d in data:
		d['y-center'] = 1 - d['y-center']
	return ret


def contrast(img, data, path):
	ret = []
	min_table = 50
	max_table = 205
	diff_table = max_table - min_table
	LUT_HC = np.arange(256, dtype = 'uint8' )
	LUT_LC = np.arange(256, dtype = 'uint8' )

	for i in range(0, min_table):
		LUT_HC[i] = 0
	for i in range(min_table, max_table):
		LUT_HC[i] = 255 * (i - min_table) / diff_table
	for i in range(max_table, 255):
		LUT_HC[i] = 255

	for i in range(256):
		LUT_LC[i] = min_table + i * (diff_table) / 255

	if random.choice([True, False]):
		high_cont_img = cv2.LUT(img, LUT_HC)
		cv2.imwrite(path+'-contH'+EXT, high_cont_img)
		ano_data_to_txt(data, path, 'contH')
		return [path+'-contH'+EXT]
	else:
		low_cont_img = cv2.LUT(img, LUT_LC)
		cv2.imwrite(path+'-contL'+EXT, low_cont_img)
		ano_data_to_txt(data, path, 'contL')
		return [path+'-contL'+EXT]


def blur(img, data, path):
	average_square = (10,10)
	blur_img = cv2.blur(img, average_square)
	cv2.imwrite(path+'-blur'+EXT, blur_img)
	ano_data_to_txt(data, path, 'blur')
	return [path+'-blur'+EXT]


def gamma(img, data, path):
	gamma1 = 0.75
	gamma2 = 1.5
	LUT_G1 = np.arange(256, dtype = 'uint8' )
	LUT_G2 = np.arange(256, dtype = 'uint8' )
	for i in range(256):
		LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
		LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

	if random.choice([True, False]):
		high_cont_img = cv2.LUT(img, LUT_G1)
		cv2.imwrite(path+'-gammaH'+EXT, high_cont_img)
		ano_data_to_txt(data, path, 'gammaH')
		return [path+'-gammaH'+EXT]
	else:
		low_cont_img = cv2.LUT(img, LUT_G2)
		cv2.imwrite(path+'-gammaL'+EXT, low_cont_img)
		ano_data_to_txt(data, path, 'gammaL')
		return [path+'-gammaL'+EXT]


def augmentation(file, img, ano_data, path_data):
	l = flip(img, ano_data, path_data)
	for arg in l:
		file.write(arg + "\n")
	if random.choice([True, False]):
		l = contrast(img, ano_data, path_data)
		for arg in l:
			file.write(arg + "\n")
	elif random.choice([True, False]):
		l = blur(img, ano_data, path_data)
		for arg in l:
			file.write(arg + "\n")
	else:
		l = gamma(img, ano_data, path_data)
		for arg in l:
			file.write(arg + "\n")



# Current directory
task_name = os.path.abspath('../darknet_v4/task_crater/')

# Directory where the data will reside, relative to 'darknet.exe'
path_data = os.path.join(task_name, 'data/')

# Percentage of images to be used for the test set
train_rate = 0.8
valid_rate = 0.2
test_rate = 1 - (train_rate + valid_rate)


# Create and/or truncate train.txt and test.txt
file_paths = glob.glob(os.path.join(const.CRATER_TRAIN_DATA, "*{}".format(EXT)))
file_paths.sort()
file_train = open(task_name + '/train.txt', 'w')  
file_val = open(task_name + '/valid.txt', 'w')
file_test = open(task_name + '/test.txt', 'w')

# Populate train.txt and test.txt
train = random.sample(file_paths, int(len(file_paths)*(train_rate+test_rate)))
test = random.sample(train, int(len(file_paths)*test_rate))
for pathAndFilename in file_paths:  
	title = os.path.splitext(os.path.basename(pathAndFilename))[0]
	path, ext = os.path.splitext(pathAndFilename)
	img, ano_data = read_data(path, ext)
	if pathAndFilename in test:
		augmentation(file_test, img, ano_data, path_data + title)
		file_test.write(pathAndFilename + "\n")
	elif pathAndFilename in train:
		augmentation(file_train, img, ano_data, path_data + title)
		file_train.write(pathAndFilename + "\n")
	else:
		augmentation(file_val, img, ano_data, path_data + title)
		file_val.write(pathAndFilename + "\n")
