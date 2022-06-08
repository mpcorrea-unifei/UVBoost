#########################################################################################################
# UVBoost: a hybrid radiative transfer and machine learning model for estimating ultraviolet radiation  # 
# Developed by Prof. Marcelo de Paula Corrêa (mpcorrea@unifei.edu.br)                                   #
# Natural Resources Institute - Federal University of Itajubá - Brazil                                  #
# Version 0.5.2                                                                                        #
# June/2022                                                                                             #
#########################################################################################################
import numpy as np
import pandas as pd
from catboost.core import CatBoostRegressor
from pathlib import Path
import timeit

print("\nWelcome to UVBoost")
print("\nUVBoost is warming up! Stay cool.... It's only a few seconds.")

#### Training set
typeset = input('\nWhat do you want to calculate?: \n1) Ultraviolet Index (Erythemal irradiances); \n2) Vitamin D weighted irradiances \n\nMy option:')
typeset = int(typeset)
start = timeit.default_timer()
df_x = pd.read_csv('x_indep_hires.csv', header=None)
if typeset == 1:
    df_y = pd.read_csv('y_dep_hires.csv', header=None)
elif typeset ==2:
    df_y = pd.read_csv('y_vitD_dep_hires.csv', header=None)
else: 
    print('\nYou should have put 1 or 2. No problem, test UVBoost anyway with some UV index calculations.')
    df_y = pd.read_csv('y_dep_hires.csv', header=None)
from sklearn.linear_model import LinearRegression
catboost = CatBoostRegressor(iterations=1000, learning_rate=0.08, depth=5, random_state=10, verbose=True)
catboost.fit(df_x,df_y)
stop = timeit.default_timer()
print(f'\nIt was not so hard! Warm-up time: {(stop - start):.1f} seconds')

inpset = input('\nSelect input data format: \n1) datasheet (csv); \n2) data on screen \n\nMy option:')
inpset = int(inpset)
while inpset:
    if inpset == 1:
        namefile = input('\nName of the input file (e.g.: input.csv)')
        df = pd.read_csv(namefile,sep=';',header=0)
        df_inp = df.iloc[:, [0,1,2]].values
#    namefile = Path(namefile).stem
        print('Input file:', namefile)
    elif inpset == 2: 
        sza = input('\nSolar Zenith Angle (in Degrees):')
        ozone = input('Ozone content (in Dobson Units):')
        aod = input('Aerosol Optical Depth (Dimensionless):')
        df_inp = np.array([sza,ozone,aod])
    else:
        print('\nYou should have put 1 or 2. No problem, test UVBoost anyway.')
        sza = input('\nSolar Zenith Angle (in Degrees):')
        ozone = input('Ozone content (in Dobson Units):')
        aod = input('Aerosol Optical Depth (Dimensionless):')
        df_inp = np.array([sza,ozone,aod])
    
    estima_CAT = catboost.predict(df_inp)

### Saving the information
    if inpset == 1:
        data = np.array([estima_CAT]).T
        data = data*40
        data
        df = pd.DataFrame(data, columns=['UVI'])
        df_inp = pd.DataFrame(df_inp, columns=['SZA','CTO','AOD'])
        df = pd.concat([df_inp,df],axis=1)
        outputfile = 'OUT_'+namefile
        df.to_csv(outputfile)
        opt = input('\nAnother calculation ? \n0: NO \nany other number: YES \n\n My option:')
        opt = int(opt)
        if opt == 0:
            exit()
    else:
        if estima_CAT < 0:
            estima_CAT = 0
        print('UVI =', estima_CAT*40)
        opt = input('\nAnother calculation ? \n0: NO \nany other number: YES \n\n My option:')
        opt = int(opt)
        if opt == 0:
            exit()