import pandas as pd
import matplotlib.pylab as plt
import glob as glob
import os as os
import nibabel as nib
import numpy as np
from pandas.plotting import scatter_matrix

df = pd.read_csv('prediction/brats_scores.csv')

plt.figure()
df['DiceCoeff'].hist()    
plt.title('Dice score for stroke data')


plt.figure()
df_sort = df.sort_values(by=['DiceCoeff'])
plt.plot(df_sort['DiceCoeff'],df_sort['Unnamed: 0'], "o", ms = 6) #, columns=list('ABCD'))
plt.title('Dice validation')


scatter_matrix(df_sort, alpha=0.5, figsize=(6, 6), diagonal='kde')    

plt.show()