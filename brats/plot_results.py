import pandas as pd
import matplotlib.pylab as plt
import glob as glob
import os as os
import nibabel as nib
import numpy as np
from pandas.plotting import scatter_matrix
from nilearn.image import load_img
import matplotlib.cm as cm

import matplotlib.animation as animation
import numpy as np
from pylab import *

prediction_dir = 'prediction' #'04_prediction_whole_img'
ext = '.png'


df = pd.read_csv(os.path.join(prediction_dir, 'brats_scores.csv'))

plt.figure()
df['DiceCoeff'].hist()    
plt.title('Dice score for stroke data')
plt.savefig(os.path.join(prediction_dir, 'dice_coef_hist' + ext))

# plot image
plt.figure()
df_sort = df.sort_values(by=['DiceCoeff'])
plt.plot(df_sort['DiceCoeff'],df_sort['Unnamed: 0'], "o", ms = 6) #, columns=list('ABCD'))
plt.title('Dice validation')
plt.savefig(os.path.join(prediction_dir, 'dice_validation' + ext))

if len(np.unique(df_sort['PredictSize'])) > 1:
    scatter_matrix(df_sort, alpha=0.5, figsize=(6, 6), diagonal='kde')    
    plt.savefig(os.path.join(prediction_dir, 'scatter_matrix' + ext))


# plot exemplar prediction set
def draw_image_masks(brain_img, true_mask, 
                    predicted_mask):
    my_cmap_predict = cm.jet
    my_cmap_predict.set_under('k', alpha=0)
    my_cmap_true = cm.PiYG
    my_cmap_true.set_under('k', alpha=0)

    plt.subplot(1,1,1)
    plt.imshow(brain_img, cmap='Greys')#, alpha = 0.5)
    im = plt.imshow(predicted_mask, cmap=my_cmap_predict, 
            interpolation='none', 
            clim=[0.9, 1], alpha = 0.5)
    im = plt.imshow(true_mask, cmap=my_cmap_true, 
            interpolation='none', 
            clim=[0.9, 1], alpha = 0.5)


plt.figure()
validation_dir = 'validation_case_365'
idx = 70

val_dir = os.path.join(prediction_dir, validation_dir)
path_t1 = os.path.join(val_dir, 'data_t1.nii.gz')
path_true = os.path.join(val_dir, 'truth.nii.gz')
path_predict = os.path.join(val_dir, 'prediction.nii.gz')
brain_img = load_img(path_t1).get_data()
true_mask = load_img(path_true).get_data()
predicted_mask = load_img(path_predict).get_data()
draw_image_masks(brain_img[:,:,idx], true_mask[:,:,idx],
                predicted_mask[:,:,idx])
plt.savefig(os.path.join(prediction_dir, validation_dir+'_mask_example' + ext))

    # make a movie
def ani_frame(prediction_dir, validation_dir = 'validation_case_365'):
    dpi = 100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(rand(300,300),cmap='gray',interpolation='nearest')
    im.set_clim([0,1])
    fig.set_size_inches([5,5])
    
    val_dir = os.path.join(prediction_dir, validation_dir)
    path_t1 = os.path.join(val_dir, 'data_t1.nii.gz')
    path_true = os.path.join(val_dir, 'truth.nii.gz')
    path_predict = os.path.join(val_dir, 'prediction.nii.gz')
    brain_img = load_img(path_t1).get_data()
    true_mask = load_img(path_true).get_data()
    predicted_mask = load_img(path_predict).get_data()

    tight_layout()


    def update_img(n):
        print('writing movie frame:',n)
        draw_image_masks(brain_img[:,:,n], true_mask[:,:,n], 
                    predicted_mask[:,:,n])
        plt.title('green: true, red: predict')
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,range(0,144,3),interval=20) #300,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    movie_path = os.path.join(prediction_dir,validation_dir+'.mp4')
    ani.save(movie_path,writer=writer,dpi=dpi)
    print('wrote a movie in:',movie_path)
    return ani

# make a movie
#ani_frame(prediction_dir=prediction_dir, validation_dir=validation_dir)
print('saved images in dir: ', prediction_dir)
plt.show()
