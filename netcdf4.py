# coding :utf-8
from netCDF4 import Dataset
import pandas as pd
import numpy as np
from numpy import concatenate

# netCDF4数据包
import os
dir = ''
files = os.listdir(dir)
for  file in files:
    dataset = Dataset('./{dir}/{file}'.format(dir=dir,file=file))

