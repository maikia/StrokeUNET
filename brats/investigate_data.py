import glob
import matplotlib.cm as cm
import matplotlib.pylab as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd


MY_CMAP_TRUE = cm.PiYG
MY_CMAP_TRUE.set_under('k', alpha=0)


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)

# find all the data files in the preprocessed directory

# plot the following info:
# scan from 3 different perspectives with mask overlayed
# give following info: image size in pixels
# overall mask size
# subject name

# plot the mean of all of the images
#
def read_data(data_dir='data/preprocessed/new_sub-r038s001'):
    scan = os.path.join(data_dir,'t1.nii.gz')
    scan_img = nib.load(os.path.abspath(scan)).get_data()
    truth = os.path.join(data_dir,'truth.nii.gz')
    truth_img = nib.load(os.path.abspath(truth)).get_data()
    return scan_img, truth_img


def plot_data(info, scan, mask, x=100, y=100, depth=100):
    depth = 150
    fig = plt.figure()
    plt.subplot(2,2,2)
    plt.imshow(scan[:, :, depth], cmap='Greys')

    im = plt.imshow(mask[:, :, depth], cmap=MY_CMAP_TRUE,
                    interpolation='none',
                    clim=[0.9, 1], alpha = 0.5)

    plt.subplot(2,2,3)
    plt.imshow(scan[:, y, :], cmap='Greys')

    im = plt.imshow(mask[:, y, :], cmap=MY_CMAP_TRUE,
                    interpolation='none',
                    clim=[0.9, 1], alpha = 0.5)

    plt.subplot(2,2,4)
    plt.imshow(scan[x, :, :], cmap='Greys')

    im = plt.imshow(mask[x, :, :], cmap=MY_CMAP_TRUE,
                    interpolation='none',
                    clim=[0.9, 1], alpha = 0.5)

    # add info about the subject
    ax = plt.subplot(2,2,1)
    ax.text(0.05, 0.9, info['name'], weight = 'bold', fontsize = 16)
    ax.text(0.05, 0.7,
             'mask shape: ' + str(info['mask_shape']).replace(',',' x'),
             fontsize = 14)
    ax.text(0.05, 0.6,
             'lesion shape: ' + str(info['lesion_shape']).replace(',',' x'),
             fontsize = 14)
    ax.text(0.05, 0.5,
             'true lesion size: ' + str(info['lesion_size']) + ' px',
             fontsize = 14)
    ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    preprocessed = os.path.join(os.getcwd(), 'data' ,'preprocessed')
    subjects = glob.glob(os.path.join(preprocessed, "*"))
    subjects = [subject for subject in subjects if os.path.isdir(subject)]

    info = {
        'name': [],
        'mask_shape': [],
        'mean_color': [],
        'lesion_shape': [],
        'lesion_size': []
    }
    info_temp = {
    }
    for idx, subj_dir in enumerate(subjects[::-1]):
        # TODO: check that it's dir (exclude files)
        fig_dir = 'figs'
        fig_dir = os.path.join(os.getcwd(), fig_dir)

        ext = '.png'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        scan_img, truth_img = read_data(data_dir = subj_dir)

        name = subj_dir.split('/')[-1]
        print('{}/{} working on: {}'.format(idx, len(subjects), name))

        info_temp['name'] = name
        info_temp['mask_shape'] = scan_img.shape
        info_temp['mean_color']=np.mean(scan_img)
        info_temp['lesion_shape']=truth_img.shape
        info_temp['lesion_size']=np.sum(truth_img)
        [info[key].append(info_temp[key]) for key in info.keys()]
        try:
            fig = plot_data(info_temp, scan_img, truth_img)
            plt.savefig(os.path.join(fig_dir, name + ext))
            plt.close('all')
        except:
            print('problem with figure')
    df_info = pd.DataFrame(info, columns = ['name', 'mask_shape', 'mean_color',
                                            'lesion_shape', 'lesion_size'])

    df_info.to_csv(os.path.join(preprocessed, 'row_info.csv'))
