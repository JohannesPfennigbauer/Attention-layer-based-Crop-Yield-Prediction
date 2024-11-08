import numpy as np
import pandas as pd



data_csv=pd.read_csv('data\soybean_data_soilgrid250.csv',delimiter=',')

data_npz=np.array(data_csv)

np.savez_compressed('data\soybean_data_compressed',data=data_npz)