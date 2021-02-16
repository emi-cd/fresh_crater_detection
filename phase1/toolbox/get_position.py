import sys, os
import pandas as pd

import download as dl
import register as rg


def get_info(path):
	base_name_origin, _ = os.path.splitext(path)
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
	return beforeID, afterID, h, w, data


def get_position(path):
	beforeID, afterID, h, w, data = get_info(path)

	i = data[data.PRODUCT_ID == '"{}"'.format(beforeID)].index
	before = rg.NacImage(data.loc[i[0]], img=False)
	i = data[data.PRODUCT_ID == '"{}"'.format(afterID)].index
	after = rg.NacImage(data.loc[i[0]], img=False)

	ret_lat = float(before.pos[8]) - (float(before.pos[8]) - float(before.pos[6])) * h / float(before.data['IMAGE_LINES'])
	ret_lon = float(before.pos[9]) + (float(before.pos[3]) - float(before.pos[9])) * w / float(before.data['LINE_SAMPLES'])
	return before, after, ret_lat, ret_lon

def main(argv):
	if len(argv) != 1:
		print('Please input path to img.')
		return 0

	before, after, ret_lat, ret_lon = get_position(argv[0], True)

	print('lat:',lat, 'lon:', lon)
	print('********** Before **********')
	print(before)
	print('********** After **********')
	print(after)


if __name__ == '__main__':
	main(sys.argv[1:])
