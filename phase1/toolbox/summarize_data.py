import pandas as pd
import glob

import get_position as gp
import const
import register as rg


def main():
	origin_df = pd.read_csv('{}crater_list.csv'.format(const.CSV_PATH), index_col=0)

	crater_col = ['CRATER_ID', 'LAT', 'LON', 'PIXEL', 'SIZE']
	crater_df = pd.DataFrame(index=[], columns=crater_col)
	crater_df.set_index('CRATER_ID')
	pair_col = ['PAIR_ID', 'CRATER_ID', 'BEFORE_ID', 'AFTER_ID', 'BEFORE_H', 'BEFORE_W']
	pair_df = pd.DataFrame(index=[], columns=pair_col)
	pair_df.set_index('PAIR_ID')
	nac_col = ['NAC_ID', 'VOLUME_ID', 'URL', 'PRODUCT_ID', 'PRODUCT_VERSION_ID', 'ORBIT_NUMBER', 
		'SLEW_ANGLE', 'MISSION_PHASE_NAME', 'DATA_QUALITY_ID', 'START_TIME', 'STOP_TIME', 'IMAGE_LINES', 
		'LINE_SAMPLES', 'SAMPLE_BITS', 'SCALED_PIXEL_WIDTH', 'SCALED_PIXEL_HEIGHT', 'RESOLUTION', 
		'EMMISSION_ANGLE', 'INCIDENCE_ANGLE', 'PHASE_ANGLE', 'NORTH_AZIMUTH', 'SUB_SOLAR_AZIMUTH', 
		'SUB_SOLAR_LATITUDE', 'SUB_SOLAR_LONGITUDE', 'SUB_SPACECRAFT_LATITUDE', 'SUB_SPACECRAFT_LONGITUDE', 
		'SOLAR_DISTANCE', 'SOLAR_LONGITUDE', 'CENTER_LATITUDE', 'CENTER_LONGITUDE', 'UPPER_RIGHT_LATITUDE', 
		'UPPER_RIGHT_LONGITUDE', 'LOWER_RIGHT_LATITUDE', 'LOWER_RIGHT_LONGITUDE', 'LOWER_LEFT_LATITUDE', 
		'LOWER_LEFT_LONGITUDE', 'UPPER_LEFT_LATITUDE', 'UPPER_LEFT_LONGITUDE', 'SPACECRAFT_ALTITUDE', 
		'TARGET_CENTER_DISTANCE', 'ORBIT_NODE', 'LRO_FLIGHT_DIRECTION']
	nac_df = pd.DataFrame(index=[], columns=nac_col)
	nac_df.set_index('NAC_ID')
	
	for index, row in origin_df.iterrows():
		new_crater_lst = glob.glob('{}{}/*/*/*.tif'.format(const.NEW_CRATERS_PATH, index))
		new_crater_lst.sort()
		FLAG = True
		
		for new_crater in new_crater_lst:
			if FLAG:
				beforeID, afterID, h, w, data = gp.get_info(new_crater)
				i = data[data.PRODUCT_ID == '"{}"'.format(beforeID)].index
				before = rg.NacImage(data.loc[i[0]], img=False)
				i = data[data.PRODUCT_ID == '"{}"'.format(afterID)].index
				after = rg.NacImage(data.loc[i[0]], img=False)

				ret_lat = float(before.pos[8]) - (float(before.pos[8]) - float(before.pos[6])) * h / float(before.data['IMAGE_LINES'])
				ret_lon = float(before.pos[9]) + (float(before.pos[3]) - float(before.pos[9])) * w / float(before.data['LINE_SAMPLES'])
				record = pd.Series([index, ret_lat, ret_lon, row.PIXEL, row.PIXEL*before.scaled_pixel_width], index=crater_df.columns)
				crater_df = crater_df.append(record, ignore_index=True)
				FLAG = False

			try:
				before_df = nac_df.query('PRODUCT_ID == \'"{}"\' '.format(before.file_name))
				before_df_id = before_df.index[0]
				print('duplicate!')
			except IndexError:
				before_data = before.data
				before_data['NAC_ID'] = len(nac_df) + 1
				before_df_id = len(nac_df) + 1
				record = pd.Series(before_data, index=nac_df.columns)
				nac_df = nac_df.append(record, ignore_index=True)
			try:
				after_df = nac_df.query('PRODUCT_ID == \'"{}"\' '.format(after.file_name))
				after_df_id = after_df.index[0]
			except IndexError:
				after_data = after.data
				after_data['NAC_ID'] = len(nac_df) + 1
				after_df_id = len(nac_df) + 1
				record = pd.Series(after.data)
				nac_df = nac_df.append(record, ignore_index=True)	
			record = pd.Series([len(pair_df)+1, index, before_df_id, after_df_id, h, w], index=pair_df.columns)
			pair_df = pair_df.append(record, ignore_index=True)
		print(index)
	crater_df.to_csv('{}crater.csv'.format(const.CSV_PATH), index=False)
	pair_df.to_csv('{}pair.csv'.format(const.CSV_PATH), index=False)
	nac_df.to_csv('{}nac.csv'.format(const.CSV_PATH), index=False)
	


if __name__ == '__main__':
	main()
