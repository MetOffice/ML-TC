import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.filters import uniform_filter as unif2D

import argparse

parser = argparse.ArgumentParser(description='Preprocess Cyclone Data')
parser.add_argument('--f', metavar='format', type=str, nargs=1, choices=['A','B','C','D','E'], help='the data format to use')
parser.add_argument('--h', metavar='hurricane', type=str, nargs=1, help='the name of the curricane to extract data from')
parser.add_argument('--v', metavar='variables', type=str, nargs='+', choices=['fg','hur','prlssn','prlst','psl','rsds','rsnds','tas','ua','va','wbpt','zg'], help='the variables to extract')
args = parser.parse_args()
print(args)
def largest_sum(a, n):
    idx = unif2D(a.astype(float),size=n, mode='constant').argmax()
    return np.unravel_index(idx, a.shape)

# zg.T3Hpoint.UMRA2T.19951123_19951126.BOB07.4p4km.nc
# zg.T3Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc
files=os.listdir()

## load hurricane 	
variables = args.v
hurricane = args.h
files=[f for f in files if '4p4km' in f and 'point' in f and hurricane in f and (v in f for v in variables)]
print(files)
input()
for variable in variables:
    wind = xr.open_dataset('../RawData/' + hurricane + '_'+ variable +'.nc')
    # print(wind)
    
    wind = wind['wind_speed_of_gust']

    all_data_points = []
    centre_data_points = []
    centre_data_points_cut = []
    # t_data_point = []
    # p_data_point = []
    # centres_cols = []
    # centres_rows = []
    valid = []
    if args.format[0]!="C":
        size = 256
    elif args.format[0]=="C":
        size=64
    side = int((size)/2)

    for data_point_time in wind.forecast_reference_time:
        for data_point_period in wind.forecast_period:
            single_data_point = wind.loc[dict(forecast_reference_time=data_point_time, forecast_period = data_point_period)]
            single_data_point = single_data_point.values
            
            (r, c) = largest_sum(single_data_point, n = 50)

            # t_data_point.append(data_point_time.values)
            # p_data_point.append(data_point_period.values)
            # centres_cols.append(c)
            # centres_rows.append(r)
            
            
            data_points_centre = single_data_point[r-side:r+side, c-side:c+side]
            centre_data_points.append(data_points_centre.copy())
                
            if args.format[0]=="D":
                single_data_point[single_data_point < 26.5] = 'NaN'
            elif args.format[0]=="E":
                single_data_point[single_data_point < 3*np.mean(single_data_point)] = 'NaN'
            back = np.zeros_like(single_data_point)
            back[:] = np.nan
            if args.format[0]!="A":
                back[r-side:r+side, c-side:c+side] = single_data_point[r-side:r+side, c-side:c+side]
            else:
                back=single_data_point
            all_data_points.append(back.copy())
            
            if args.format[0]!="A":
                data_points_centre = single_data_point[r-side:r+side, c-side:c+side]
            else:
                data_points_centre = single_data_point
            # back[r-side:r+side, c-side:c+side] = data_points_centre
            #filter out empty points
            if np.count_nonzero(~np.isnan(data_points_centre)) > 3000 and (args.format[0]=="A" or (len(data_points_centre)==size and len(data_points_centre[0])==size)):
                centre_data_points_cut.append(data_points_centre)
                
    print(len(centre_data_points_cut))
    ## save the results
    np.savez("/project/ciid/projects/ML-TC/Data/"+hurricane+"_"+args.format[0]+".npz",centre_data_points_cut)
