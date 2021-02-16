import sys, os
import pandas as pd
import glob

import register as rg
import download as dl
import const


def format(file_path):
	base_name_origin, _ = os.path.splitext(file_path)
	base_name = base_name_origin.split('/')
	lat = base_name[-3].split('-')[0]
	if len(base_name[-3].split('-')) == 3:
		lat = '-' + base_name[-3].split('-')[1]
		lon = base_name[-3].split('-')[2]
	else:
		lat = base_name[-3].split('-')[0]
		lon = base_name[-3].split('-')[1]
	beforeID = base_name[-2]
	afterID = base_name[-1].split('-')[2]
	h = int(base_name[-1].split('-')[0])
	w = int(base_name[-1].split('-')[1])

	data = dl.get_data_from_point([lat, lon])
	i = data[data.PRODUCT_ID == '"{}"'.format(beforeID)].index
	dl.download_nac_one(data, i[0])
	before = rg.NacImage(data.loc[i[0]])
	i = data[data.PRODUCT_ID == '"{}"'.format(afterID)].index
	dl.download_nac_one(data, i[0])
	after = rg.NacImage(data.loc[i[0]])

	pair = rg.TemporalPair(before, after)
	pair.make_dif(output=False)
	pair.extract(w, h, base_name_origin)
	if os.path.exists(base_name_origin + '.png') and os.path.exists(base_name_origin + '.tif'):
		os.remove(base_name_origin + '.png')


def main():
	# New crater
	new_craters = glob.glob(const.NEW_CRATERS_PATH + '*/*/*.png')
	for file_path in new_craters:
		try:
			format(file_path)
			print(file_path)
		except Exception as e:
			print(e)
	print('new_craters: ', len(new_craters))


	# Splotches
	# unclear_new_craters = glob.glob(const.UNCLEAR_NEW_CRATERS_PATH + '*/*/*.png')
	# for file_path in unclear_new_craters:
	# 	try:
	# 		format(file_path)
	# 		print(file_path)
	# 	except Exception as e:
	# 		print(e)
	# print('unclear_new_craters: ', len(unclear_new_craters))

	# Else
	# els = glob.glob('/hdd_mount/ELSE/*/*/*.png')
	# for file_path in els:
	# 	format(file_path)
	# 	print(file_path)
	# print('els: ', len(els))

	# path = '/hdd_mount/OUTPUT/-25.73-351.98/M1136613257LC/5250-3750-M1182515611RC.png'
	# format(path)
	

if __name__ == '__main__':
	main()
