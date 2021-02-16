import sys, os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

import download as dl
import register as rg
import const

IMAGE_SIZE = 2000


def make_multi_tiff(imgs, output_path):
	stack = []
	for img in imgs:
		tmp_img = rg.convert_to_uint8(img)
		stack.append(Image.fromarray(255 - tmp_img))
	stack[0].save('{}.tiff'.format(output_path), compression="tiff_deflate", save_all=True, append_images=stack[1:])


def save_as_png(imgs, names, output_path):
	for i, img in enumerate(imgs):
		tmp_img = rg.convert_to_uint8(img)
		cv2.imwrite('{}_{}.png'.format(output_path, names[i]), tmp_img)


def clip_from_img(nac, lat, lon, image_size=IMAGE_SIZE):
	pts1 = np.float32([nac.pos[2:4][::-1],nac.pos[4:6][::-1], nac.pos[6:8][::-1], nac.pos[8:][::-1]])
	nac_lines = nac.file.label['IMAGE']['LINES']
	nac_line_samples = nac.file.label['IMAGE']['LINE_SAMPLES']
	pts2 = np.float32([[nac_line_samples-nac.left_padding, 0],[nac_line_samples-nac.left_padding, nac_lines],\
				[0-nac.left_padding, nac_lines],[0-nac.left_padding, 0]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	pts = M.dot([lon, lat, 1])
	pts = [pts[0]/pts[2], pts[1]/pts[2]]

	x = int(pts[0])
	y = int(pts[1])
	y_bottom = y-image_size//2 if y-image_size//2 > 0 else 0
	y_top = y+image_size//2 if y+image_size//2 < nac.image.shape[0] else nac.image.shape[0]
	x_bottom = x-image_size//2 if x-image_size//2 > 0 else 0
	x_top = x+image_size//2 if x+image_size//2 < nac.image.shape[1] else nac.image.shape[1]
	print(y_bottom, y_top, x_bottom, x_top)
	img = nac.image[y_bottom:y_top, x_bottom:x_top]
	return img


def collect_after_img(df, crater_df, pair_df, nac_df, image_size=IMAGE_SIZE):
	ret = []
	for crater_id in df.index.values:
		print('crater_id: ', crater_id)
		lat = float(crater_df.at[crater_id, 'LAT'])
		lon =  float(crater_df.at[crater_id, 'LON'])
		data = dl.get_data_from_point([lat, lon])
		this_pair_df = pair_df.query('CRATER_ID == {}'.format(crater_id))
		l = np.unique(this_pair_df.AFTER_ID.values)
		after_nac = []
		for nac_id in l:
			after_date = nac_df.at[nac_id, 'STOP_TIME']
			after_nac.append(after_date)
		after_nac.sort()
		if len(after_nac) == 0:
			return ret
		can_df = data[data['STOP_TIME'] >= after_nac[0]]

		imgs = []
		names = []
		for nac_id in can_df.head().index.values:
			dl.download_nac_one(data, nac_id)
			nac = rg.NacImage(data.loc[nac_id])
			imgs.append(clip_from_img(nac, lat, lon, image_size))
			names.append(nac.file_name)
		output_path = os.path.join(const.BIG_CRATER, '{}'.format(crater_id))
		ret.append([imgs, names, output_path])
	return ret


def collect_after_img_size(crater_size):
	crater_df = pd.read_csv('{}crater.csv'.format(const.CSV_PATH), index_col=0)
	pair_df = pd.read_csv('{}pair.csv'.format(const.CSV_PATH), index_col=0)
	nac_df = pd.read_csv('{}nac.csv'.format(const.CSV_PATH), index_col=0)

	big_crater_df = crater_df[crater_df['SIZE'] > crater_size]
	for i in range(len(big_crater_df)):
		data = collect_after_img(big_crater_df.iloc[[i]], crater_df, pair_df, nac_df)
		for imgs, names, output_path in data:
			save_as_png(imgs, names, output_path)


def collect_after_img_sig(sig=1):
	crater_df = pd.read_csv('{}crater_sig.csv'.format(const.CSV_PATH), index_col=0)
	pair_df = pd.read_csv('{}pair.csv'.format(const.CSV_PATH), index_col=0)
	nac_df = pd.read_csv('{}nac.csv'.format(const.CSV_PATH), index_col=0)

	significant_crater_df = crater_df[crater_df.SIGNIFICANT > sig]
	for i in range(len(significant_crater_df)):
		data = collect_after_img(significant_crater_df.iloc[[i]], crater_df, pair_df, nac_df)
		for imgs, names, output_path in data:
			save_as_png(imgs, names, output_path)


def collect_after_img_sig_only(crater_id, image_size=IMAGE_SIZE):
	crater_df = pd.read_csv('{}crater_sig.csv'.format(const.CSV_PATH), index_col=0)
	pair_df = pd.read_csv('{}pair.csv'.format(const.CSV_PATH), index_col=0)
	nac_df = pd.read_csv('{}nac.csv'.format(const.CSV_PATH), index_col=0)
	return  collect_after_img(crater_df[crater_id-1:crater_id], crater_df, pair_df, nac_df, image_size)


if __name__ == '__main__':
	args = sys.argv
	# main(args[1:])
	main()