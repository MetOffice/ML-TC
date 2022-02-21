import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.filters import uniform_filter as unif2D

import argparse

vars={'fg':'wind_speed_of_gust','hur':'relative_humidity','psl':'air_pressure_at_sea_level','rsds':'surface_downwelling_shortwave_flux_in_air','rsnds':'net_down_surface_sw_flux_corrected','tas':'air_temperature','ua':'x_wind','va':'y_wind','wbpt':'wet_bulb_potential_temperature','zg':'geopotential_height'}

parser = argparse.ArgumentParser(description='Preprocess Cyclone Data')
parser.add_argument('--f', metavar='format', type=str, nargs=1, choices=['A','B','C','D','E'], help='the data format to use')
parser.add_argument('--h', metavar='hurricane', type=str, nargs=1, help='the name of the hurricane to extract data from')
parser.add_argument('--p', metavar='pressure', type=int, default=200,nargs='?', help='the pressure level to use where needed')
parser.add_argument('--v', metavar='variables', type=str, nargs='+', choices=['fg','hur','psl','rsds','rsnds','tas','ua','va','wbpt','zg'], help='the variables to extract')
parser.add_argument('--e', metavar='ensembles', type=str, nargs=1, choices=['separate','joint'], help='whether to mix the ensembles or keep them separate (ex. to maintain time order')
args = parser.parse_args()
print(args)
def largest_sum(a, n):
    idx = unif2D(a.astype(float),size=n, mode='constant').argmax()
    return np.unravel_index(idx, a.shape)

# zg.T3Hpoint.UMRA2T.19951123_19951126.BOB07.4p4km.nc
# zg.T3Hpoint.UMRA2T.19910428_19910501.BOB01.4p4km.nc
files=os.listdir('/projects/metoffice/ml-tc/ML-TC/RawData/')
#print(files)
## load hurricane 	
variables = args.v
hurricane = args.h[0]
files=[f for f in files if '4p4km' in f and 'point' in f and hurricane in f and any(v in f for v in variables)]
#print(files)
for variable in variables:
    wind = xr.open_dataset('/projects/metoffice/ml-tc/ML-TC/RawData/'+files[0])
    # print(wind)
    wind = wind[vars[variable]]

    all_data_points = []
    centre_data_points = []
    centre_data_points_cut = []
    # t_data_point = []
    # p_data_point = []
    # centres_cols = []
    # centres_rows = []
    valid = []
    if args.f[0]!="C":
        size = 256
    elif args.f[0]=="C":
        size=64
    side = int((size)/2)


    for data_point_time in wind.forecast_reference_time:
        if args.e[0]=='separate':
            centre_data_points_cut.append([])
            #print(centre_data_points_cut)
        for data_point_period in wind.forecast_period:
            if variable in ['hur','va','wbpt','zg']:
                single_data_point = wind.loc[dict(pressure=args.p,forecast_reference_time=data_point_time, forecast_period = data_point_period)]
            else: 
                single_data_point = wind.loc[dict(forecast_reference_time=data_point_time, forecast_period = data_point_period)]
            single_data_point = single_data_point.values
            #print(single_data_point.shape) 
            (r, c) = largest_sum(single_data_point, n = 50)
            
            data_points_centre = single_data_point[r-side:r+side, c-side:c+side]
            centre_data_points.append(data_points_centre.copy())
                
            if args.f[0]=="D":
                single_data_point[single_data_point < 26.5] = 'NaN'
            elif args.f[0]=="E":
                single_data_point[single_data_point < 3*np.mean(single_data_point)] = 'NaN'
            back = np.zeros_like(single_data_point)
            back[:] = np.nan
            if args.f[0]!="A":
                back[r-side:r+side, c-side:c+side] = single_data_point[r-side:r+side, c-side:c+side]
            else:
                back=single_data_point
            all_data_points.append(back.copy())
            
            if args.f[0]!="A":
                data_points_centre = single_data_point[r-side:r+side, c-side:c+side]
            else:
                data_points_centre = single_data_point
            # back[r-side:r+side, c-side:c+side] = data_points_centre
            #filter out empty points
            if np.count_nonzero(~np.isnan(data_points_centre)) > 3000 and (args.f[0]=="A" or (len(data_points_centre)==size and len(data_points_centre[0])==size)):
                if args.e[0]=='joint':
                    centre_data_points_cut.append(data_points_centre)
                else:
                    #print(data_points_centre.shape)
                    centre_data_points_cut[-1].append(np.array(data_points_centre))
                    #print(type(data_points_centre)) 
                    #print(len(centre_data_points_cut))
                    #print(len(centre_data_points_cut[0]))
                    #print(len(centre_data_points_cut[-1]))
                
    ## save the results
    print("sizes")
    print(len(centre_data_points_cut))
    print(len(centre_data_points_cut[0]))
    #print(len(centre_data_points_cut[0][0]))
    print(centre_data_points_cut[0][0].shape)
    if args.e[0] == "separate":
        centre_data_points_cut=np.array(centre_data_points_cut,dtype="object")
    np.savez("/projects/metoffice/ml-tc/ML-TC/Data/"+hurricane+"_"+args.f[0]+"_"+variable+"_"+args.e[0]+".npz",centre_data_points_cut)
    
