import matplotlib.pylab as plt
import os
import pandas as pd
import pickle
import numpy as np


ext = '.png'

# training_df = pd.read_csv("./training_isenee.log").set_index('epoch')
training_df = pd.read_csv("./training.log").set_index('epoch')
plt.figure()
plt.plot(training_df['loss'].values, label='training loss')
plt.plot(training_df['val_loss'].values, label='validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim((0, len(training_df.index)))
plt.legend(loc='upper right')
plt.savefig('loss_graph'+ext)


training_ids = pickle.load( open( "training_ids.pkl", "rb" ) )
validation_ids = pickle.load( open( "validation_ids.pkl", "rb" ) )
print('training ids: ', np.sort(training_ids))
print('validation ids:', np.sort(validation_ids))

plt.figure()
plt.plot(-training_df['dice_coefficient'].values, label='dice_coefficient')
plt.plot(-training_df['val_dice_coefficient'].values,
         label='val_dice_coefficient')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim((0, len(training_df.index)))
plt.legend(loc='upper right')
plt.savefig('coefficient_'+ext)
plt.show()
