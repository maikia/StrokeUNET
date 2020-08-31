import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pylab as plt
from nilearn.image import load_img
import numpy as np
import os as os
import pandas as pd


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
def ani_frame(prediction_dir, validation_dir='subject_16'):
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
    path_t1 = os.path.join(val_dir, 'data_no_skull_norm_t1.nii.gz')
    path_true = os.path.join(val_dir, 'truth.nii.gz')
    path_predict = os.path.join(val_dir, 'prediction.nii.gz')
    brain_img = load_img(path_t1).get_fdata()
    true_mask = load_img(path_true).get_fdata()
    predicted_mask = load_img(path_predict).get_fdata()

    plt.tight_layout()

    def update_img(n):
        print('writing movie frame:', n)
        draw_image_masks(brain_img[:, :, n], true_mask[:, :, n],
                         predicted_mask[:, :, n])
        plt.title('green: true, red: predict')
        return im

    ani = animation.FuncAnimation(fig, update_img, range(0, 128, 3))
    # interval=20)
    writer = animation.writers['ffmpeg'](fps=5)  # fps: frames per second

    movie_path = os.path.join(prediction_dir, validation_dir+'.mp4')
    ani.save(movie_path, writer=writer, dpi=dpi)

    plt.close('all')
    print('wrote a movie in:', movie_path)
    return ani


def plot_dice_coeff_score_hist(df_scores, prediction_dir, ext):
    plt.figure()
    df_scores['DiceCoeff'].hist()
    plt.title('Dice score for stroke data')
    plt.savefig(os.path.join(prediction_dir, 'dice_coef_hist' + ext))


def plot_dice_coeff_score(df_scores, prediction_dir, ext):
    plt.figure(figsize=(10, 20))

    df_sort = df_scores.sort_values(by=['DiceCoeff'])
    plt.plot(df_sort['DiceCoeff'], df_sort['Unnamed: 0'], "o", ms=6)
    plt.title('Dice validation')
    plt.savefig(os.path.join(prediction_dir, 'dice_validation' + ext))


def plot_for_subject(prediction_dir, subject_dir, depth_idx=70,
                     filename_t1='data_no_skull_norm_t1.nii.gz',
                     filename_truth='truth.nii.gz',
                     filename_predict='prediction.nii.gz', ext='.png'):
    plt.figure()

    val_dir = os.path.join(prediction_dir, subject_dir)
    path_t1 = os.path.join(val_dir, filename_t1)
    path_true = os.path.join(val_dir, filename_truth)
    path_predict = os.path.join(val_dir, filename_predict)
    brain_img = load_img(path_t1).get_fdata()
    true_mask = load_img(path_true).get_fdata()
    predicted_mask = load_img(path_predict).get_fdata()
    draw_image_masks(brain_img[:, :, depth_idx], true_mask[:, :, depth_idx],
                     predicted_mask[:, :, depth_idx])
    plt.savefig(os.path.join(prediction_dir,
                             subject_dir+'_mask_example' + ext))


def plot_for_all_subjects(prediction_dir, depth_idx=70,
                          filename_t1='data_no_skull_norm_t1.nii.gz',
                          filename_truth='truth.nii.gz',
                          filename_predict='prediction.nii.gz', ext='.png'):
    # get all the validation dirs
    dir_iterator = next(os.walk(prediction_dir))[1]

    for idx, subject_dir in enumerate(dir_iterator):
        plot_for_subject(prediction_dir, subject_dir, depth_idx=depth_idx,
                         ext=ext,
                         filename_t1=filename_t1,
                         filename_truth=filename_truth,
                         filename_predict=filename_predict)
    print(f'saved result plots for {idx+1} subjects')


def _make_new_folder(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == "__main__":
    prediction_dir = 'prediction_126_cut_size'
    ext = '.png'
    filename_t1 = 'data_no_skull_norm_t1.nii.gz'
    filename_truth = 'truth.nii.gz'
    filename_predict = 'prediction.nii.gz'

    # plot image
    df = pd.read_csv(os.path.join(prediction_dir, 'brats_scores.csv'))
    plot_dice_coeff_score_hist(df, prediction_dir, ext=ext)
    plot_dice_coeff_score(df, prediction_dir, ext=ext)
    plot_for_all_subjects(prediction_dir, ext=ext,
                          filename_t1=filename_t1,
                          filename_truth=filename_truth,
                          filename_predict=filename_predict)

    # make a movie
    ani_frame(prediction_dir, validation_dir='subject_16')

    print('saved images in dir: ', prediction_dir)

    plt.show()
