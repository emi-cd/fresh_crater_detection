import sys, os, glob, shutil
import pandas as pd
import numpy as np
import time
from multiprocessing import Process, Pool

import download as dl
import register as rg
import const


def cleaning(path):
	for b in glob.glob(path+'M*'):
		try: 
			os.rmdir(b)
		except:
			continue
	if len(glob.glob(path+'*')) == 1:
		shutil.rmtree(path)


def calc(data, key, val, before, index, lat, lon, output_path, file_name):
	print(key, val)
	after = rg.NacImage(data.loc[val])
	start_ = time.time()
	pair = rg.TemporalPair(before, after)
	pair.make_dif(output_path, True)
	elapsed_time_ = time.time() - start_
	new_row = pd.DataFrame([index, lat, lon, elapsed_time_, \
				data.at[key, 'PRODUCT_ID'], data.at[val, 'PRODUCT_ID'],\
				data.at[key, 'START_TIME'], data.at[val, 'START_TIME'],\
				pair.detect_num, pair.all_num]).T
	new_row.to_csv(file_name, mode='a', header=False, index=False)
	print('Finish', key, val, ':', elapsed_time_)


def main(argv):
	if len(argv) != 2:
		print('Please input 1 args. \nex) python [path to crater file] [path to done list file]')
		return 0


	points_df = pd.read_csv(argv[0])
	done_df = pd.read_csv(argv[1])
	for index, row in points_df.iterrows():
		if row['PROG']:
			continue

		point_origin = [row['LAT'], row['LON']]
		lats = [point_origin[0]-1+i for i in range(3)]
		lons = [point_origin[1]-1+i*0.1 for i in range(21)]
		i_done_df = done_df.query('origin_id == {}'.format(index))
		did = []

		for lon in lons:
			for lat in lats:
				df =  i_done_df.query('lat == {} and lon == {}'.format(lat, lon))
				if len(df) > 0:
					continue
				start = time.time()

				# Prepare the data
				point = [lat, lon]
				point_path = '{}-{}/'.format(lat, lon)
				output_path = const.OUTPUT_PATH + point_path
				os.makedirs(output_path, exist_ok=True)
				print(point)

				data = dl.get_data_from_point(point)
				pair = dl.make_temporal_pair(data)
				data.to_csv(output_path + 'INFO.csv')

				nacs = list(pair.keys())
				for t in list(pair.values()):
					nacs += t
				nacs = list(set(nacs))
				process_list = []
				for nac in nacs:
					process = Process(
						target=dl.download_nac_one,
						kwargs={'data': data, 'i': nac})
					process.start()
					process_list.append(process)
				for process in process_list:
					process.join()

				for key, vals in pair.items():
					print(key, vals)
					before = rg.NacImage(data.loc[key])
					process_list = []
					for val in vals:
						if not (data.loc[key, 'PRODUCT_ID'], data.loc[val, 'PRODUCT_ID']) in did:
							did.append((data.loc[key, 'PRODUCT_ID'], data.loc[val, 'PRODUCT_ID']))
							process = Process(
								target=calc,
								kwargs={'data': data, 'key': key, 'val':val, 'before':before,\
									'index': index, 'lat': lat, 'lon':lon, 'output_path':output_path,\
									'file_name':argv[1]})
							process.start()
							process_list.append(process)
					for process in process_list:
						process.join()
				cleaning(output_path)

				
				elapsed_time = time.time() - start
				print('elapsed_time:{} [sec]\n\n'.format(elapsed_time))

		points_df.iloc[index, 2] = True
		points_df.to_csv(argv[0], index=False)
		print('*********************************************')


if __name__ == '__main__':
	args = sys.argv
	# args = ['test.csv', 'done_list.csv']
	main(args[1:])