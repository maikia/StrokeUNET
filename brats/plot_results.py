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
#from pylab import *

prediction_dir = 'prediction' #'04_prediction_whole_img'
#image_dir = ''
ext = '.png'


df = pd.read_csv(os.path.join(prediction_dir, 'brats_scores.csv'))

plt.figure()
df['DiceCoeff'].hist()    
plt.title('Dice score for stroke data')
plt.savefig(os.path.join(prediction_dir, 'dice_coef_hist' + ext))

# plot image
plt.figure(figsize=(10,20))

df_sort = df.sort_values(by=['DiceCoeff'])
#plt.rc('ytick', labelsize=2) 
plt.plot(df_sort['DiceCoeff'],df_sort['Unnamed: 0'], "o", ms = 6) #, columns=list('ABCD'))
plt.title('Dice validation')
plt.savefig(os.path.join(prediction_dir, 'dice_validation' + ext))

if len(np.unique(df_sort['PredictSize'])) > 1:
    scatter_matrix(df_sort, alpha=0.5, figsize=(6, 6), diagonal='kde')    
    plt.savefig(os.path.join(prediction_dir, 'scatter_matrix' + ext))


def draw_all_images(prediction_dir, brain_depth=72, ext='.png'):
    """
    prediction_dir:
    brain_depth: at which pxl depth the image of the brain will be drawn
    """
    # get all the validation dirs
    dir_iterator = next(os.walk(prediction_dir))[1]
    my_cmap_predict = cm.jet
    my_cmap_predict.set_under('k', alpha=0)
    my_cmap_true = cm.PiYG
    my_cmap_true.set_under('k', alpha=0)

    for val_dir in dir_iterator:
        path_prediction = os.path.join(prediction_dir, val_dir)
        path_img_save = os.path.join(prediction_dir, val_dir)

        path_t1 = os.path.join(val_dir, 'data_t1.nii.gz')
        path_true = os.path.join(val_dir, 'truth.nii.gz')
        path_predict = os.path.join(val_dir, 'prediction.nii.gz')

        # save image
        brain_img = load_img(path_t1).get_data()[:,:,brain_depth]
        validation_case = val_dir.split("_")[-1]

        # brain image
        plt.figure()
        plt.subplot(1,1,1)
        plt.imshow(brain_img, cmap='Greys')
        plt.title('validation_case: '+ validation_case)
        brain_name = os.path.join(path_img_save, 'the_brain'+ext)
        plt.savefig(brain_name)

        # brain image with mask
        mask_image = load_img(path_true).get_data()
        mask_at_depth = mask_image[:,:,brain_depth]
        #mask_image_all = 
        im = plt.imshow(mask_image, cmap=my_cmap_true, 
            interpolation='none', 
            clim=[0.9, 1], alpha = 0.5)
        plt.title('lesion size: '+ validation_case)
        truth_name = os.path.join(path_img_save, 'truth_on_brain'+ext)
        plt.savefig(truth_name)



        plt.close("all")

    # draw brain with truth mask
    # title: lesion, pixel size: <path to the file>

    # draw brain with predicted mask
    # title: predicted lesion, pixel size:

    # draw mask overlap
    # title size of overlap:, score:

def make_tex_code_to_present_predicted_results():
    pass

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
validation_dir = 'subject_22'
idx = 70

val_dir = os.path.join(prediction_dir, validation_dir)
path_t1 = os.path.join(val_dir, 'data_no_skull_norm_t1.nii.gz')
path_true = os.path.join(val_dir, 'truth.nii.gz')
path_predict = os.path.join(val_dir, 'prediction.nii.gz')
brain_img = load_img(path_t1).get_data()
true_mask = load_img(path_true).get_data()
predicted_mask = load_img(path_predict).get_data()
draw_image_masks(brain_img[:, :, idx], true_mask[:, :, idx],
                 predicted_mask[:, :, idx])
plt.savefig(os.path.join(prediction_dir, validation_dir+'_mask_example' + ext))

    # make a movie
def ani_frame(prediction_dir, validation_dir='validation_case_365'):
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

#draw_all_images(prediction_dir, brain_depth=100, ext='.png')

print('saved images in dir: ', prediction_dir)

plt.show()
