import pandas as pd
import matplotlib.pylab as plt
import glob as glob
import os as os
import nibabel as nib
import numpy as np
from pandas.plotting import scatter_matrix

prediction_dir = '03_prediction'
ext = '.png'


df = pd.read_csv(os.path.join(prediction_dir, 'brats_scores.csv'))

plt.figure()
df['DiceCoeff'].hist()    
plt.title('Dice score for stroke data')
plt.savefig(os.path.join(prediction_dir, 'dice_coef_hist' + ext))


plt.figure()
df_sort = df.sort_values(by=['DiceCoeff'])
plt.plot(df_sort['DiceCoeff'],df_sort['Unnamed: 0'], "o", ms = 6) #, columns=list('ABCD'))
plt.title('Dice validation')
plt.savefig(os.path.join(prediction_dir, 'dice_validation' + ext))

scatter_matrix(df_sort, alpha=0.5, figsize=(6, 6), diagonal='kde')    
plt.savefig(os.path.join(prediction_dir, 'scatter_matrix' + ext))

plt.show()