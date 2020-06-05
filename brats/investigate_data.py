import matplotlib.pylab as plt
import nibabel as nib
import numpy as np
import os


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


def plot_data(info, scan):
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(scan[:,:,72], cmap='Greys')
    ax = plt.subplot(2,2,4)

    ax.text(0.05, 0.9, info['name'], weight = 'bold', fontsize = 16)
    ax.text(0.05, 0.8,
             'shape: ' + str(info['shape']).replace(',',' x'),
             fontsize = 14)
    ax.text(0.05, 0.75,
             'true lesion size: ' + str(info['lesion_size']) + ' px',
             fontsize = 14)
    ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    subj_dir = 'data/preprocessed/new_sub-r038s001'
    scan_img, truth_img = read_data(data_dir = subj_dir)

    info = {
        'name': subj_dir.split('/')[-1],
        'shape': scan_img.shape,
        'lesion_size': np.sum(truth_img)
    }
    fig = plot_data(info, scan_img)  #, truth_img)

    plt.show()