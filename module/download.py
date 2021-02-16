import sys, os
import subprocess
import pandas as pd
import numpy as np
import time
import urllib.request
from multiprocessing import Process

import const


def get_data(cmd):
	'''
	Get data from tool.

	Parameters
	----------
	cmd : str
		Execute this command.

	Returns
	-------
	data : dataframe
		Result as dataframe type
	'''
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
	data = []
	for line in proc.stdout:
		data.append(line.decode()[:-1].split('\t'))
	data = pd.DataFrame(data[1:], columns=data[0])
	return data


def get_data_from_point(point):
	'''
	Get the nac image at location point.

	Parameters
	----------
	point : list of float
		Latitude and longitude.

	Returns
	-------
	data : dataframe
		Information of the nac that contain the point.
	'''
	cmd = 'PATH/TO/TOOL --database PATH/TO/DB -p -- {} {}'.format(point[0], point[1])
	data = get_data(cmd.split(' '))
	data['RESOLUTION'] = data['RESOLUTION'].astype(float)
	data['EMMISSION_ANGLE'] = data['EMMISSION_ANGLE'].astype(float)
	data['INCIDENCE_ANGLE'] = data['INCIDENCE_ANGLE'].astype(float)
	data['PHASE_ANGLE'] = data['PHASE_ANGLE'].astype(float)
	data['NORTH_AZIMUTH'] = data['NORTH_AZIMUTH'].astype(float)
	data['SUB_SOLAR_AZIMUTH'] = data['SUB_SOLAR_AZIMUTH'].astype(float)
	return data


def get_data_from_point_rectangle(point, r_lat, r_lon):
	'''
	Get the nac image at location point.

	Parameters
	----------
	point : list of float
		Latitude and longitude.
	r_lat : float
		Error of latitude.
	r_lon : float
		Error of longitude.

	Returns
	-------
	data : dataframe
		Information of the nac that contain the point.
	'''
	cmd = 'PATH/TO/TOOL --database PATH/TO/DB -r -- {} {} {} {}'.format(point[1]-r_lon, point[1]+r_lon, point[0]-r_lat, point[0]+r_lat)
	data = get_data(cmd.split(' '))
	data['RESOLUTION'] = data['RESOLUTION'].astype(float)
	data['EMMISSION_ANGLE'] = data['EMMISSION_ANGLE'].astype(float)
	data['INCIDENCE_ANGLE'] = data['INCIDENCE_ANGLE'].astype(float)
	data['PHASE_ANGLE'] = data['PHASE_ANGLE'].astype(float)
	data['NORTH_AZIMUTH'] = data['NORTH_AZIMUTH'].astype(float)
	data['SUB_SOLAR_AZIMUTH'] = data['SUB_SOLAR_AZIMUTH'].astype(float)
	return data


def make_temporal_pair(data):
	'''
	Make temporal pair from point as this function name.

	Parameters
	----------
	data : dataframe
		Nac images that meet the conditions.

	Returns
	-------
	ret: dictionary
		{id: [id, ...], id: [id, ...]}
		This is temporal pair.
	'''
	data = data[data.INCIDENCE_ANGLE < 50]
	data = data[data.RESOLUTION < 1.75]
	ret = {}
	for i in data.index:
		emmission_angle = data.at[i, 'EMMISSION_ANGLE']
		incident_angle = data.at[i, 'INCIDENCE_ANGLE']
		phase_angle = data.at[i, 'PHASE_ANGLE']
		# resolution = data.at[i, 'RESOLUTION']
		north_azimuth = data.at[i, 'NORTH_AZIMUTH']
		sub_solar_azimuth = data.at[i, 'SUB_SOLAR_AZIMUTH']
		date = data.at[i, 'START_TIME']
		# FROM 2020/4/29
		q = '-10<EMMISSION_ANGLE-{0}<10 and -3<=INCIDENCE_ANGLE-{1}<=3 and\
				-10<PHASE_ANGLE-{2}<10 and -10<SUB_SOLAR_AZIMUTH-{4}<10 \
				'.format(emmission_angle, incident_angle, phase_angle, north_azimuth, \
					sub_solar_azimuth)
		# q = '-15<EMMISSION_ANGLE-{0}<15 and -3<=INCIDENCE_ANGLE-{1}<=3 and\
		# 		-15<PHASE_ANGLE-{2}<15 and -15<=NORTH_AZIMUTH-{3}<=15 and\
		# 		-15<SUB_SOLAR_AZIMUTH-{4}<15 \
		# 		'.format(emmission_angle, incident_angle, phase_angle, north_azimuth, \
		# 			sub_solar_azimuth)
		# q = '-15<EMMISSION_ANGLE-{0}<15 and -3<=INCIDENCE_ANGLE-{1}<=3 and\
		# 		-15<PHASE_ANGLE-{2}<15 \
		# 		-15<=(NORTH_AZIMUTH-SUB_SOLAR_AZIMUTH)-({3}-{4})<=15\
		# 		'.format(emmission_angle, incident_angle, phase_angle, north_azimuth, \
		# 			sub_solar_azimuth)
		match = data.query(q)
		match_lst = [j for j in match.index.tolist() if i < j]
		match_lst = [j for j in match_lst if data.at[j, 'START_TIME'] != data.at[i, 'START_TIME']]
		if len(match_lst) > 0:
			ret[data.loc[i].name] = match_lst
	return ret


def make_temporal_pair_with_date(data, date):
	'''
	Make temporal pair from point as this function name.
	I use this function for serching of impact flash craters.

	Parameters
	----------
	data : dataframe
		Nac images that meet the conditions.
	date : str
		Before image of temporal pair should be before date and after image should be
		after date.

	Returns
	-------
	ret: dictionary
		{id: [id, ...], id: [id, ...]}
		This is temporal pair.
	'''
	# data = data[data.INCIDENCE_ANGLE < 50]
	ret = {}
	
	data['STOP_TIME'] = data['STOP_TIME'].map(lambda x: x.replace('"', '').split(' ')[0])
	data['STOP_TIME'] = pd.to_datetime(data['STOP_TIME'])
	before = data[data['STOP_TIME'] < date]
	after = data[data['STOP_TIME'] > date]
	if len(before) == 0 or len(after) == 0:
		return ret

	after = after.head(1)
	after_data = np.array([after.at[after.index[0], 'EMMISSION_ANGLE'],
							after.at[after.index[0], 'INCIDENCE_ANGLE'],
							after.at[after.index[0], 'PHASE_ANGLE'],
							# after.at[after.index[0], 'RESOLUTION'],
							# after.at[after.index[0], 'NORTH_AZIMUTH'],
							# after.at[after.index[0], 'SUB_SOLAR_AZIMUTH']
							])

	min_sa = -1
	for i in before.index:	
		before_data = np.array([before.at[i, 'EMMISSION_ANGLE'],
								before.at[i, 'INCIDENCE_ANGLE'],
								before.at[i, 'PHASE_ANGLE'],
								# before.at[i, 'RESOLUTION'],
								# before.at[i, 'NORTH_AZIMUTH'],
								# before.at[i, 'SUB_SOLAR_AZIMUTH']
								])

		sa = ((after_data - before_data)**2).mean()
		if sa < min_sa or min_sa < 0:
			min_sa = sa
			ret[i] = [after.index[0]]

	return ret


def make_temporal_pair_with_date_all(data, date):
	'''
	Make temporal pair from point as this function name.
	There are no restrictions.

	Parameters
	----------
	data : dataframe
		Nac images that meet the conditions.
	date : str
		Before image of temporal pair should be before date and after image should be
		after date.

	Returns
	-------
	ret: dictionary
		{id: [id, ...], id: [id, ...]}
		This is temporal pair.
	'''
	# data = data[data.INCIDENCE_ANGLE < 50]
	ret = {}
	
	data['STOP_TIME'] = data['STOP_TIME'].map(lambda x: x.replace('"', '').split(' ')[0])
	data['STOP_TIME'] = pd.to_datetime(data['STOP_TIME'])
	before = data[data['STOP_TIME'] < date]
	after = data[data['STOP_TIME'] > date]
	if len(before) == 0 or len(after) == 0:
		return ret

	for b in list(before.index):
		print(b)
		ret[b] = list(after.index)

	return ret


def download_nac_from_url(url, file_name):
	'''
	Download *.IMG from url.
	Waring; of cource, you can use request library, but it often stops. So I use wget.

	Parameters
	----------
	url : str
		Using this url when doenload.
	file_name : str
		Save file as this file name.
	'''
	proc = subprocess.run(['wget', '-t', '5', url, '-P', const.NAC_IMAGE_PATH])
							# stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	print('Download: ', file_name)


def download_nac(data, pair):
	'''
	Download *.IMG of pair.

	Parameters
	----------
	data : dataframe
		Data frame containing nac information.
	pair : str
		{id: [id, ...], id: [id, ...]}
		This is temporal pair.
	'''
	ret = []
	for i in pair.keys():
		ret.append(i)
		ret.extend(pair[i])
	ret = list(set(ret))
	for i in ret:
		file_name = data.at[i, 'PRODUCT_ID']
		file_name = file_name.replace('"', '')
		nac_paths = const.NAC_IMAGE_PATH.split(';')
		for path in nac_paths:
			if os.path.exists(path + file_name + '.IMG'):
				break
		else:			
			url = data.at[i, 'URL']
			url = url.replace('"', '')
			print('Download: ', file_name)
			download_nac_from_url(url, file_name)


def download_nac_one(data, i):
	'''
	Download *.IMG.

	Parameters
	----------
	data : dataframe
		Data frame containing nac information.
	i : int
		Download nac of data[i].
	'''
	file_name = data.at[i, 'PRODUCT_ID']
	file_name = file_name.replace('"', '')
	nac_paths = const.NAC_IMAGE_PATH.split(';')
	for path in nac_paths:
		if os.path.exists(path + file_name + '.IMG'):
			break
	else:
		url = data.at[i, 'URL']
		url = url.replace('"', '')
		print('Download: ', file_name)
		download_nac_from_url(url, file_name)


def download_nac_all(data_origin):
	'''
	Download all *.IMG in data_origin

	Parameters
	----------
	data_origin : dataframe
		Data frame containing nac information.
	'''
	urls = []
	for index, data in data_origin.iterrows():
		file_name = data['PRODUCT_ID']
		file_name = file_name.replace('"', '')
		nac_paths = const.NAC_IMAGE_PATH.split(';')
		for path in nac_paths:
			if os.path.exists(path + file_name + '.IMG'):
				break
		else:
			url = data['URL']
			url = url.replace('"', '')
			# download_nac_from_url(url, file_name)
			urls.append((url, file_name))
	size = 5
	for i in range(0, len(urls), size):
		process_list = []
		for j in range(size):
			if len(urls) > (i+j):
				process = Process(
					target=download_nac_from_url,
					kwargs={'url': urls[i+j][0], 'file_name':urls[i+j][1]})
				process.start()
				process_list.append(process)
		for process in process_list:
			process.join()


def main(argv):
	if len(argv) != 2:
		return 0
	point = [argv[0], argv[1]]
	print('point:', point)
	data = get_data_from_point(point)
	pair = make_temporal_pair(data)
	download_nac(data, pair)
	return 0


if __name__ == '__main__':
	args = sys.argv
	main(args[1:])