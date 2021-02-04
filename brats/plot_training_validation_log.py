import matplotlib.pylab as plt
import os
import pandas as pd


training_df = pd.read_csv("./training.log").set_index('epoch')
plt.figure()
plt.plot(training_df['loss'].values, label='training loss')
plt.plot(training_df['val_loss'].values, label='validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim((0, len(training_df.index)))
plt.legend(loc='upper right')
# plt.savefig(os.path.join(prediction_dir, 'loss_graph'+ext))
plt.show()
