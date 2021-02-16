import sys, os
import pandas as pd
import glob

import register as rg
import download as dl
import const


def get_trans_img_16(file_path):
	base_name_origin, _ = os.path.splitext(file_path)
	base_name = base_name_origin.split('/')
	beforeID = base_name[-2]
	afterID = base_name[-1].split('-')[2]
	if len(base_name[-3].split('-')) == 3:
		lat = '-' + base_name[-3].split('-')[1]
		lon = base_name[-3].split('-')[2]
	else:
		lat = base_name[-3].split('-')[0]
		lon = base_name[-3].split('-')[1]
	h = int(base_name[-1].split('-')[0])
	w = int(base_name[-1].split('-')[1])

	data = dl.get_data_from_point([lat, lon])
	i = data[data.PRODUCT_ID == '"{}"'.format(beforeID)].index
	before = rg.NacImage(data.loc[i[0]])
	i = data[data.PRODUCT_ID == '"{}"'.format(afterID)].index
	after = rg.NacImage(data.loc[i[0]])
	pair = rg.TemporalPair(before, after)
	pair.make_dif(output=False)
	imgs = pair.extract(w, h, base_name_origin, save=False)
	return imgs[1]


def format(file_path):
	base_name_origin, _ = os.path.splitext(file_path)
	base_name = base_name_origin.split('/')
	lat = base_name[3].split('-')[0]
	if len(base_name[3].split('-')) == 3:
		lat = '-' + base_name[3].split('-')[1]
		lon = base_name[3].split('-')[2]
	else:
		lat = base_name[3].split('-')[0]
		lon = base_name[3].split('-')[1]
	beforeID = base_name[4]
	afterID = base_name[5].split('-')[2]
	h = int(base_name[5].split('-')[0])
	w = int(base_name[5].split('-')[1])

	data = dl.get_data_from_point([lat, lon])
	i = data[data.PRODUCT_ID == '"{}"'.format(beforeID)].index
	before = rg.NacImage(data.loc[i[0]])
	i = data[data.PRODUCT_ID == '"{}"'.format(afterID)].index
	after = rg.NacImage(data.loc[i[0]])

	pair = rg.TemporalPair(before, after)
	pair.make_dif(output=False)
	pair.extract(w, h, base_name_origin)
	if os.path.exists(base_name_origin + '.png'):
		os.remove(base_name_origin + '.png')


