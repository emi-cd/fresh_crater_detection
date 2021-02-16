import pandas as pd
import glob
import os

import collect_nac as cn
import const
import download as dl
import register as rg


def main():
	# cn.collect_after_img_size(crater_size=10.0)
	# for imgs, output_path in data:
	# 	cn.save_as_png(imgs, output_path)

	crater_df = pd.read_csv('{}crater_sig.csv'.format(const.CSV_PATH), index_col=0)
	pair_df = pd.read_csv('{}pair.csv'.format(const.CSV_PATH), index_col=0)
	nac_df = pd.read_csv('{}nac.csv'.format(const.CSV_PATH), index_col=0)

	significant_crater_df = crater_df[crater_df.SIGNIFICANT > 1]
	for crater_id in significant_crater_df.index.values:
		file_path = glob.glob('{}{}/*/*/*.tif'.format(const.NEW_CRATERS_PATH, crater_id))

		base_name_origin, _ = os.path.splitext(file_path[0])
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

		img = pair.trans[h:h+608, w:w+608]
		# img = cn.clip_from_img(nac, float(lat), float(lon), image_size=608)
		cn.save_as_png([img], [afterID], '{}{}'.format(const.CRATER_TRAIN_DATA, crater_id))


if __name__ == '__main__':
	main()