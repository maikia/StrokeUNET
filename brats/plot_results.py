import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pylab as plt
from nilearn.image import load_img
import numpy as np
import os as os
import pandas as pd
from pandas.plotting import scatter_matrix

PREDICTION_DIR = 'prediction'
EXT = '.png'


def draw_all_images(prediction_dir, brain_depth=72,
                    filname_t1='data_no_skull_norm_t1.nii.gz',
                    filename_truth='truth.nii.gz',
                    filename_predict='prediction.nii.gz', ext='.png'):
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
        path_img_save = os.path.join(prediction_dir, val_dir)

        path_t1 = os.path.join(val_dir, filname_t1)
        path_true = os.path.join(val_dir, filename_truth)
        # path_predict = os.path.join(val_dir, filename_predict)

        # save image
        brain_img = load_img(path_t1).get_data()[:, :, brain_depth]
        validation_case = val_dir.split("_")[-1]

        # brain image
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.imshow(brain_img, cmap='Greys')
        plt.title('validation_case: ' + validation_case)
        brain_name = os.path.join(path_img_save, 'the_brain'+ext)
        plt.savefig(brain_name)

        # brain image with mask
        mask_image = load_img(path_true).get_data()
        mask_at_depth = mask_image[:, :, brain_depth]
        plt.imshow(mask_at_depth, cmap=my_cmap_true,
                   interpolation='none',
                   clim=[0.9, 1], alpha=0.5)
        plt.title('lesion size: ' + validation_case)
        truth_name = os.path.join(path_img_save, 'truth_on_brain'+ext)
        plt.savefig(truth_name)

        plt.close("all")


# plot exemplar prediction set
def draw_image_masks(brain_img, true_mask, predicted_mask):
    my_cmap_predict = cm.jet
    my_cmap_predict.set_under('k', alpha=0)
    my_cmap_true = cm.PiYG
    my_cmap_true.set_under('k', alpha=0)

    plt.subplot(1, 1, 1)
    plt.imshow(brain_img, cmap='Greys')  # , alpha = 0.5)
    plt.imshow(predicted_mask, cmap=my_cmap_predict,
               interpolation='none',
               clim=[0.9, 1], alpha=0.5)
    plt.imshow(true_mask, cmap=my_cmap_true,
               interpolation='none',
               clim=[0.9, 1], alpha=0.5)


# make a movie
def ani_frame(prediction_dir, validation_dir='validation_case_365'):
    dpi = 100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(np.random.rand(300, 300), cmap='gray',
                   interpolation='nearest')
    im.set_clim([0, 1])
    fig.set_size_inches([5, 5])

    val_dir = os.path.join(prediction_dir, validation_dir)
    path_t1 = os.path.join(val_dir, 'data_t1.nii.gz')
    path_true = os.path.join(val_dir, 'truth.nii.gz')
    path_predict = os.path.join(val_dir, 'prediction.nii.gz')
    brain_img = load_img(path_t1).get_data()
    true_mask = load_img(path_true).get_data()
    predicted_mask = load_img(path_predict).get_data()

    plt.tight_layout()

    def update_img(n):
        print('writing movie frame:', n)
        draw_image_masks(brain_img[:, :, n], true_mask[:, :, n],
                         predicted_mask[:, :, n])
        plt.title('green: true, red: predict')
        return im

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, range(0, 144, 3),
                                  interval=20)  # 300,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    movie_path = os.path.join(prediction_dir, validation_dir+'.mp4')
    ani.save(movie_path, writer=writer, dpi=dpi)
    print('wrote a movie in:', movie_path)
    return ani


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(PREDICTION_DIR, 'brats_scores.csv'))

    plt.figure()
    df['DiceCoeff'].hist()
    plt.title('Dice score for stroke data')
    plt.savefig(os.path.join(PREDICTION_DIR, 'dice_coef_hist' + EXT))

    # plot image
    plt.figure(figsize=(10, 20))

    df_sort = df.sort_values(by=['DiceCoeff'])

    plt.plot(df_sort['DiceCoeff'], df_sort['Unnamed: 0'], "o", ms=6)
    plt.title('Dice validation')
    plt.savefig(os.path.join(PREDICTION_DIR, 'dice_validation' + EXT))

    if len(np.unique(df_sort['PredictSize'])) > 1:
        scatter_matrix(df_sort, alpha=0.5, figsize=(6, 6), diagonal='kde')
        plt.savefig(os.path.join(PREDICTION_DIR, 'scatter_matrix' + EXT))

    plt.figure()
    validation_dir = 'subject_19'
    idx = 70

    val_dir = os.path.join(PREDICTION_DIR, validation_dir)
    path_t1 = os.path.join(val_dir, 'data_no_skull_norm_t1.nii.gz')
    path_true = os.path.join(val_dir, 'truth.nii.gz')
    path_predict = os.path.join(val_dir, 'prediction.nii.gz')
    brain_img = load_img(path_t1).get_data()
    true_mask = load_img(path_true).get_data()
    predicted_mask = load_img(path_predict).get_data()
    draw_image_masks(brain_img[:, :, idx], true_mask[:, :, idx],
                     predicted_mask[:, :, idx])
    plt.savefig(os.path.join(PREDICTION_DIR,
                             validation_dir+'_mask_example' + EXT))
    # make a movie
    # ani_frame(prediction_dir=prediction_dir, validation_dir=validation_dir)

    # draw_all_images(prediction_dir, brain_depth=100, ext='.png')

    print('saved images in dir: ', PREDICTION_DIR)

    plt.show()
