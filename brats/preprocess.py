"""
Tools for converting, normalizing, and fixing the T1 brain scans and
corresponding lesion data.
"""
import os
import shutil
import subprocess
import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from nilearn import plotting
from nilearn.image import load_img, math_img, new_img_like
from nipype.interfaces.fsl import BET


def find_dirs(raw_dir='data/', ext='.nii.gz'):
    """searches for the directories within the given directory and up the tree
    which contain at least one file with a given extension.
    ----------
    raw_dir : string or path to the directory to be searched
    ext : string, extension to the searched for files, eg '.png' :
    Returns
    -------
    path_list
        list of the directories which contain at least a single file with `ext`
        extension
    """
    print(f'Wait. I am searching for "{ext}" files in {raw_dir}')
    path_list = []
    for dirname, dirnames, filenames in os.walk(raw_dir):
        # save directories with all filenames ending on `ext`
        for filename in filenames:
            if filename[-len(ext):] == ext:
                path_list.append(dirname)
                break
    return path_list


def apply_mask_to_image(mask, img):
    """given a mask (in a form of nilearn image) it applies it to the img and
        returns the masked img. That is: all the values equals to 0 in the mask
        will now be equal to 0 in the masked. The shape of the data remains the
        same.
    ----------
    mask : nilearn image, binary
    img : nilearn image
    Returns
    -------
    masked
        img masked with a mask
    """
    assert mask.shape == img.shape
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()
    img_data[mask_data == 0] = 0
    masked = new_img_like(img, img_data, affine=None, copy_header=False)
    return masked


def strip_skull_mask(t1_file_in, t1_file_out, mask_file_out):
    """strips the skull from the T1 MRI image
    ----------
    t1_file_in: an existing file name
        path nifti file with the T1 MRI image with the skull
    t1_file_out: path to the file name
        path where the new image with skull stripped is to be saved
    mask_file_out : path to the file name
        path where the calculated mask used to strip the t1_file_in image is
        to be saved

    Returns
    -------
    t_img: nilearn image
        t1 with the skull stripped of
    mask: nilearn image
        the calculated mask
    """
    skullstrip = BET(in_file=t1_file_in, out_file=t1_file_out, mask=False)
    skullstrip.run()

    # it sets all the values > 0 to 1 creating a mask
    t_img = load_img(t1_file_out)
    mask = math_img('img > 0', img=t_img)
    mask.to_filename(mask_file_out)

    return t_img, mask


def combine_lesions(path, lesion_str='Lesion'):
    """it loads all the images from the files found in the given path which
       include lesion_str in their name (assumed to be the lesion files), adds
       them together, and sets all the values different from 0 to 1.
    ----------
    path: path to the directory
        path where the lesion files are stored
    lesion_str: string
        string which must be included in the name of the lesion file

    Returns
    -------
    t_img: image (binary) or 0
        returns 0 if there were no matching files found or the nilearn image as
        the combined lesion file
    """

    n_lesions = 0
    print('combining lesions and setting them to 0s and 1s')
    for file_name in os.listdir(path):
        if lesion_str in file_name:
            # is lesion
            lesion_img = load_img(os.path.join(path, file_name))
            lesion_data = lesion_img.get_fdata()
            if n_lesions == 0:
                lesion = lesion_data
            else:
                lesion += lesion_data
            n_lesions += 1
    if n_lesions > 0:
        lesion[lesion > 0] = 1
        masked = new_img_like(lesion_img, lesion,
                              affine=None, copy_header=False)
        return masked
    else:
        # there are no lesions found
        warnings.warn('there are no lesion files with name including '
                      f'{lesion_str} found in the {path}.')
        return 0


def find_file(path, include_str='t1', exclude_str='lesion'):
    """finds all the files in the given path which include include_str in their
       name and do not include exclude_str
    ----------
    path: path to the directory
        path where the files are stored
    include_str: string
        string which must be included in the name of the file
    exclude_str: strin
       string which may not be included in the name of the file

    Returns
    -------
    files: list
        list of filenames matching the given criteria
    """
    files = os.listdir(path)
    if include_str is not None:
        files = [n_file for n_file in files if (include_str in n_file)]
    if exclude_str is not None:
        files = [n_file for n_file in files if (exclude_str not in n_file)]
    return files


def clean_all(dir_to_clean):
    """removes all the files and directories from the given path
    ----------
    path: dir_to_clean
        path to directory to be cleaned out
    """

    if os.path.exists(dir_to_clean):
        shutil.rmtree(dir_to_clean)
    os.mkdir(dir_to_clean)
    print(f'cleaned all from {dir_to_clean}')


def init_base(path, column_names, file_name='subject_info.csv'):
    """initites the .csv file with the correct column names if it does not
       already exist. Checks for the latest subject id and returns the next
       subject id which should be used
    ----------
    path: path to the directory
        path to the directory where the .csv file should be stored
    column_names: list of strings
        names of the columns which will be set in the top of the file if it
        does not already exist
    file_name: string
        name of the .csv file, ending on .csv. Eg. subjects.csv


    Returns
    -------
    subj_id: int
        next subject id which should be used
    dfObj: pandas dataframe
        the content of the csv file
    """

    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
        dfObj = pd.DataFrame(columns=column_names)
        dfObj.to_csv(file_path)
        return 1, dfObj
    else:
        dfObj = pd.read_csv(file_path)
        if len(dfObj) == 0:
            return 1, dfObj
        else:
            subj_id = np.max(dfObj['NewID']) + 1
            return subj_id, dfObj


def init_dict(key_names, **kwargs):
    """initiates new dictionary with the keys set to key_names, values either
       None, or specified in kwargs
    ----------
    key_names: path to the directory
        path to the directory where the .csv file should be stored
    **kwargs: any
        values will be set in the dictionary. Keys should match keys from the
        key_names

    Returns
    -------
    dict: dictionary
        dictionary with values set to either None or as specified by kwargs
    """

    next_subj = dict.fromkeys(column_names, None)
    for key, value in kwargs.items():
        next_subj[key] = value
    return next_subj


def normalize_to_mni(t1_in, t1_out, template, matrix_out):
    """transforms the t1 to the same space as template.
    ----------
    t1_in: path to the nifti file
       path to the file to be normalized
    t1_out: path
       where the transformed t1 image should be saved
    template: path
       to the template brain image
    matrix_out: path
       path where the matrix representing the transformation should be saved
    """
    subprocess.run([
            "flirt",
            "-in", t1_in,
            "-out", t1_out,
            "-ref", template_out,
            "-omat", matrix_out])


def normalize_to_transform(t1_in, t1_out, template_out, matrix_in):
    """normalizes the T1 image to the given tranformation.
    ----------
    t1_in: path to the nifti file
       path to the file to be normalized
    t1_out: path
       where the transformed t1 image should be saved
    matrix_in: path
       path to the matrix used for transformation

    Returns
    -------
    dict: dictionary
        dictionary with values set to either None or as specified by kwargs
    """
    # takes the saved matrix_out and uses it to transform lesion_in and saves
    # the tranformed lesion_in under lesion_out
    subprocess.run([
                    "flirt",
                    "-in", t1_in,
                    "-out", t1_out,
                    "-applyxfm", "-init", matrix_in,
                    "-ref", template_out])

    # converts mask to binary. The higher threshold the smaller the mask
    subprocess.run([
                    "fslmaths", t1_out,
                    "-thr", "0.5",
                    "-bin", t1_out])


if __name__ == "__main__":
    # loop through available images
    #   unify all the available masks to a single mask with only 1s and 0s
    #   remove the skull (from t1 and mask)
    #   move all the images (t1 and masks) to mni space
    #   noramalize images (same average color) ??
    column_names = ['RawPath', 'ProcessedPath', 'NewID',
                                      'RawSize_x', 'RawSize_y', 'RawSize_z',
                                      'NewSize_x', 'NewSize_y', 'NewSize_z',
                                      'RawLesionSize', 'NewLesionSize']
    ### first data set
    dataset = {
        "raw_dir": '../../data/ATLAS_R1.1/',
        "lesion_str": 'Lesion',
        "t1_inc_str": 't1',
        "t1_exc_str": None
    }
    # second data set
    dataset2 = {
        "raw_dir": '../../data/BIDS_lesions_zip/',
        "lesion_str": 'lesion',
        "t1_inc_str": 'T1',
        "t1_exc_str": 'label'
    }
    data = dataset

    raw_dir_healthy = '../../data/healthy'
    results_dir = 'data/preprocessed/'
    ext = '.png'
    # careful, if set to True, all the previous preprocessing saved
    # data will be removed
    rerun_all = True  # careful !!

    # find other mni templates at:
    # http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
    template_out = os.path.join('../../data/',
                                'mne_template',
                                'mni_icbm152_t1_tal_nlin_asym_09c.nii.gz')

    path_list = find_dirs(raw_dir=data['raw_dir'], ext='.nii.gz')
    n_dirs = len(path_list)

    if rerun_all:
        clean_all(results_dir)
    csv_file = 'subject_info.csv'
    next_id, df_info = init_base(results_dir, column_names=column_names,
                                 file_name=csv_file)

    for idx, path_raw in enumerate(path_list):
        print(f'{idx+1}/{n_dirs}, subject {next_id}, working on {path_raw}')
        path_results = os.path.join(results_dir, f'subject_{next_id}')
        path_figs = os.path.join(path_results, 'figs')

        # create output path (TODO: for now if sth goes wrong this path remains
        os.mkdir(path_results)
        os.mkdir(path_figs)

        # initiates info dict for the new subject
        next_subj = init_dict(column_names, RawPath=path_raw,
                              ProcessedPath=path_results, NewID=next_id)

        # check if multiple lesion files are saved
        # combines them and sets to 0 or 1
        lesion_img = combine_lesions(path_raw, lesion_str=data['lesion_str'])
        next_subj['RawLesionSize'] = int(np.sum(lesion_img.get_fdata()))
        next_subj['RawSize_x'], next_subj['RawSize_y'], \
            next_subj['RawSize_z'] = lesion_img.shape

        # remove the skull (from t1 and mask)
        print('stripping skull')
        file_in = find_file(path=path_raw,
                            include_str=data['t1_inc_str'],
                            exclude_str=data['t1_exc_str'])
        assert len(file_in) == 1  # only a single T1 file should be found
        t1_file = os.path.join(path_raw, file_in[0])
        t1_no_skull_file = os.path.join(path_results, 't1_no_skull.nii.gz')
        mask_no_skull_file = os.path.join(path_results, 'mask_no_skull.nii.gz')
        # import pdb; pdb.set_trace()
        # correct_bias(t1_file, 'temp.nii.gz')

        no_skull_t1_img, mask_img = strip_skull_mask(
            t1_file, t1_no_skull_file, mask_no_skull_file)
        no_skull_lesion_img = apply_mask_to_image(mask_img, lesion_img)

        # save files, plot images (?)
        no_skull_lesion_file = os.path.join(path_results,
                                            'no_skull_lesion.nii.gz')
        # no_skull_t1_file = os.path.join(path_results, 'no_skull_t1.nii.gz ')
        no_skull_lesion_img.to_filename(no_skull_lesion_file)

        assert no_skull_lesion_img.shape == no_skull_t1_img.shape

        # align the image
        print('normalizing to mni space')
        no_skull_norm_t1_file = os.path.join(
            path_results, 'no_skull_norm_t1.nii.gz'
        )
        no_skull_norm_lesion_file = os.path.join(
            path_results, 'no_skull_norm_lesion.nii.gz')

        transform_matrix_file = os.path.join(path_results, 'matrix.mat')

        normalize_to_mni(t1_no_skull_file, no_skull_norm_t1_file, template_out,
                         transform_matrix_file)  # uses flirt from fsl
        normalize_to_transform(no_skull_lesion_file, no_skull_norm_lesion_file,
                         template_out, transform_matrix_file)

        print('plot template')
        plotting.plot_stat_map(template_out,
                               title='template', display_mode='ortho', dim=-1,
                               draw_cross=False, annotate=False, bg_img=None)
        plt.savefig(os.path.join(path_figs, '0_template' + ext))
        print('plotting original image')
        plotting.plot_stat_map(t1_file,
                               title='original', display_mode='ortho', dim=-1,
                               draw_cross=False, annotate=False, bg_img=None)
        plt.savefig(os.path.join(path_figs, '1_original_image' + ext))

        print('plotting mask')
        plotting.plot_stat_map(mask_no_skull_file,
                               title='mask', display_mode='ortho', dim=-1,
                               draw_cross=False, annotate=False, bg_img=None)
        plt.savefig(os.path.join(path_figs, '2_mask_no_skull' + ext))

        print('plotting t1, no skull')
        plotting.plot_stat_map(t1_no_skull_file,
                               title='original, no skull',
                               display_mode='ortho', dim=-1, draw_cross=False,
                               annotate=False, bg_img=None)
        plt.savefig(os.path.join(path_figs, '3_original_no_skull' + ext))

        print('plotting original and maskedlesion')
        plotting.plot_stat_map(lesion_img,
                               title='lesion', display_mode='ortho', dim=-1,
                               draw_cross=False, annotate=False, bg_img=None)
        plt.savefig(os.path.join(path_figs, '4_lesion' + ext))
        plotting.plot_stat_map(no_skull_lesion_img,
                               title='lesion, mask', display_mode='ortho',
                               dim=-1, draw_cross=False, annotate=False,
                               bg_img=None)
        plt.savefig(os.path.join(path_figs, '5_mask_lesion_no_skull' + ext))

        print('plotting normalized images')
        plotting.plot_stat_map(no_skull_norm_t1_file,
                               title='t1, no skull, norm',
                               display_mode='ortho', dim=-1,
                               draw_cross=False, annotate=False, bg_img=None)
        plt.savefig(os.path.join(path_figs, '6_t1_no_skull_norm' + ext))

        plotting.plot_stat_map(no_skull_norm_lesion_file,
                               title='lesion, no skull, norm',
                               display_mode='ortho', dim=-1,
                               draw_cross=False, annotate=False, bg_img=None)
        plt.savefig(os.path.join(path_figs, '7_lesion_no_skull_norm' + ext))

        plotting.plot_roi(lesion_img, bg_img=t1_file, title="before",
                          draw_cross=False, cmap='autumn')
        plt.savefig(os.path.join(path_figs, '8_before_t1_lesion' + ext))

        plotting.plot_roi(no_skull_norm_lesion_file,
                          bg_img=no_skull_norm_t1_file, title="after",
                          draw_cross=False, cmap='autumn')
        plt.savefig(os.path.join(path_figs, '9_after_t1_lesion' + ext))
        plt.close('all')

        no_skull_norm_lesion_img = load_img(no_skull_norm_lesion_file)
        no_skull_norm_lesion_data = no_skull_norm_lesion_img.get_fdata()
        next_subj['NewLesionSize'] = int(np.sum(no_skull_norm_lesion_data))

        no_skull_norm_t1_img = load_img(no_skull_norm_t1_file)

        next_subj['NewSize_x'], next_subj['NewSize_y'], \
            next_subj['NewSize_z'] = no_skull_norm_lesion_data.shape
        df = pd.DataFrame(next_subj, index=[next_id])
        df.to_csv(os.path.join(results_dir, csv_file), mode='a', header=False)
        next_id += 1
        # do we want to resample image?
        # resampled_stat_img = resample_to_img(file_in, template)
        # do we need to retreshold to make it binary mask again?
        # resampled_mask = resample_to_img(mask, template) # clip = bool (for mask)


        # add the new image files to the path assigning the number to it
        # add the new image files to the .csv files (old path, new path, old
        # patient name, new patient name, lesion size before preprocessing,
        # add the picture of the transformation
        # add the parameter to generate from the beginning or add to already
        # existing
        # try to speed up the process
        # lesion size after preprocessing, image size)
        # print('\n')
        # if not err:
        #    path_list.remove(path)
        #    continue

    # TODO: correct saving healthy data
    # TODO: look at all the data, all the scans if they look alright
    # TODO: for now no normalization, rescale is done and the bias is not
    # corrected, do we want it here?
    data_dir = '../../data/ATLAS_R1.1/'
    # data_dir_new = '../../data/BIDS_lesions_zip/'
    # healthy_data_dir = '../../data/healthy'

    output_data_dir = 'data/preprocessed/'

    convert_data_old(stroke_folder=data_dir, out_folder=output_data_dir)
    # convert_data_new(stroke_folder=data_dir_new, out_folder=output_data_dir)

    # make essential preprocessing steps:
    # 1. if eough info and if needed: magnetic field inhomogeneity correction
    # using FSL topup command
    # 2. align to a standard space like MNI: (scans and lesions)
    # 3. remove the skull


    # TODO:
    # add option if no lesion: healthy patient. create empty mask
    #
    # more?
    # - shall we be correcting bias?
    # - shall we be normalizing images?
    #
    # add notes:
    # Note1: there might be more than one mask stored for one patients. Those
    # masks will be collapsed into one
    # Note2: some patients have scans at multiple times. Those scans will be
    # considered as separate patient
    # the output image masks will always consist only of [0, 1]s
