"""
Tools for converting, normalizing, and fixing the T1 brain scans and
corresponding lesion data.
"""
import os
import shutil
import subprocess
import warnings

from joblib import Memory, Parallel, delayed
import matplotlib.pylab as plt
from nilearn import plotting
from nilearn.image import load_img, math_img, new_img_like
from nipype.interfaces.fsl import BET
import numpy as np
import pandas as pd
from unet3d.utils.utils import find_dirs

if os.environ.get('DISPLAY'):
    N_JOBS = 1
else:
    # running on the server
    N_JOBS = -1

mem = Memory('./')


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


def strip_skull_mask(t1_file_in, t1_file_out, mask_file_out, frac=0.3):
    """strips the skull from the T1 MRI image
    ----------
    t1_file_in: an existing file name
        path nifti file with the T1 MRI image with the skull
    t1_file_out: path to the file name
        path where the new image with skull stripped is to be saved
    mask_file_out: path to the file name
        path where the calculated mask used to strip the t1_file_in image is
        to be saved
    frac: 'auto' or float
        fractional intensity threshold, default is 'auto' (note: different than
        in BET, where it is 0.5). If frac is 'auto', then the mean of the
        t1_file_in is calculated. For mean < 20, frac is set to 0.5. If mean is
        > 20 and < 25 then frac is set to 0.4. If mean is > 25 then frac is set
        to 0.3.
        TODO: the correctness of those settings should be tested on the
        larger dataset

    Returns
    -------
    t_img: nilearn image
        t1 with the skull stripped of
    mask: nilearn image
        the calculated mask
    """
    if frac == 'auto':
        data = load_img(t1_file_in).get_fdata()
        md = np.mean(data)
        if md < 20:
            frac = 0.5
        elif md < 25:
            frac = 0.4
        else:
            frac = 0.5
    skullstrip = BET(in_file=t1_file_in, out_file=t1_file_out, mask=False,
                     frac=frac)
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

    if not os.environ.get('DISPLAY'):
        subprocess.run([
                "nice", "-n", "10",
                "flirt",
                "-in", t1_in,
                "-out", t1_out,
                "-ref", template,
                "-omat", matrix_out])
    else:
        subprocess.run([
                "flirt",
                "-in", t1_in,
                "-out", t1_out,
                "-ref", template,
                "-omat", matrix_out])


def normalize_to_transform(t1_in, t1_out, template, matrix_in):
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
    if not os.environ.get('DISPLAY'):
        subprocess.run([
                    "nice", "-n", "10",
                    "flirt",
                    "-in", t1_in,
                    "-out", t1_out,
                    "-applyxfm", "-init", matrix_in,
                    "-ref", template])

    # converts mask to binary. The higher threshold the smaller the mask
        subprocess.run([
                    "nice", "-n", "10",
                    "fslmaths", t1_out,
                    "-thr", "0.5",
                    "-bin", t1_out])
    else:
        subprocess.run([
                    "flirt",
                    "-in", t1_in,
                    "-out", t1_out,
                    "-applyxfm", "-init", matrix_in,
                    "-ref", template])

        # converts mask to binary. The higher threshold the smaller the mask
        subprocess.run([
                    "fslmaths", t1_out,
                    "-thr", "0.5",
                    "-bin", t1_out])


def read_dataset(name):
    """reads the info for the dataset with the name
       Note1: sometimes there is more than one lesion stored for one patients.
       Those lesions will be collapsed into one

       Note2: some patients have scans at multiple times. Those scans will be
       considered as separate patient the output image masks will always
       consist only of [0, 1]s
    ----------
    name: string
       name of the dataset to use

    Returns
    -------
    dict: dictionary
        dictionary info for that dataset
    """
    # first data set
    dataset1 = {
        "name": 'dataset_1',
        "raw_dir": '../../data/ATLAS_R1.1/',
        "lesion_str": 'Lesion',
        "t1_inc_str": 't1',
        "t1_exc_str": None
    }
    if name == dataset1['name']:
        return dataset1

    # second data set
    dataset2 = {
        "name": 'dataset_2',
        "raw_dir": '../../data/BIDS_lesions_zip/',
        "lesion_str": 'lesion',
        "t1_inc_str": 'T1',
        "t1_exc_str": 'label'
    }
    if name == dataset2['name']:
        return dataset2

    # third dataset (healthy patinets)
    # here all the scans are in the single directory
    dataset3 = {
        "name": 'dataset_healthy',
        "raw_dir": '../../data/healthy/',
        "lesion_str": 'lesion',
        "t1_inc_str": 'T1',
        "t1_exc_str": 'label'
    }
    if name == dataset3['name']:
        return dataset3
    return None


def bias_field_correction(t1_in):
    """corrects field bias using fast method from fsl.
       It will save multiple nifti files in the directory (as described by FAST
       https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST#Fast) where t1_in is
       stored, however only the path to the biased corrected image will be
       returned
    ----------
    t1_in: path to the nifti file
       path to the file to be biased corrected
    Returns
    -------
    out_file: path to the nifti file
        path to the file biased corrected
    """
    basename = 'bias'
    out_dir = os.path.dirname(t1_in)
    basename = os.path.join(out_dir, basename)
    if not os.environ.get('DISPLAY'):
        subprocess.run([
                "nice", "-n", "10",
                "fast",
                "-t", "1",  # is T1
                "-o", basename,  # basename for outputs
                "-B",  # output restored image (bias-corrected image)
                t1_in])  # nifti image to bias correct
    else:
        subprocess.run([
            "fast",
            "-t", "1",  # is T1
            "-o", basename,  # basename for outputs
            "-B",  # output restored image (bias-corrected image)
            t1_in])  # nifti image to bias correct

    out = basename + '_restore.nii.gz'
    return out


def plot_t1(path_t1, title, fig_dir, fig_file):
    plotting.plot_stat_map(path_t1, title=title,
                           display_mode='ortho', dim=-1,
                           draw_cross=False, annotate=False, bg_img=None,
                           cmap='Greys_r',
                           cut_coords=(0, 0, 0))
    plt.savefig(os.path.join(fig_dir, fig_file))


def plot_mask(path_mask, title, fig_dir, fig_file):
    plotting.plot_stat_map(path_mask, title=title,
                           display_mode='ortho', dim=-1,
                           draw_cross=False, annotate=False, bg_img=None,
                           cmap='autumn_r',
                           cut_coords=[0, 0, 0])
    plt.savefig(os.path.join(fig_dir, fig_file))


def plot_overlay(path_mask, path_bg, title, fig_dir, fig_file):
    plotting.plot_roi(path_mask, bg_img=path_bg, title=title,
                      draw_cross=False, cmap='autumn',
                      cut_coords=[0, 0, 0])
    plt.savefig(os.path.join(fig_dir, fig_file))


def preprocess_image(next_id, path_raw, path_template):
    print(f'subject {next_id}, working on {path_raw}')
    path_results = os.path.join(results_dir, f'subject_{next_id}')
    path_figs = os.path.join(path_results, 'figs')

    # create output path. if it already exists. remove it and create clean
    if os.path.exists(path_results):
        shutil.rmtree(path_results)
    os.mkdir(path_results)
    os.mkdir(path_figs)

    # initiates info dict for the new subject
    next_subj = init_dict(column_names, RawPath=path_raw,
                          ProcessedPath=path_results, NewID=next_id)

    # 1. combine lesions
    # check if multiple lesion files are saved
    # combines them and sets to 0 or 1
    print(f's{next_id}: combining lesions and setting them to 0s and 1s')
    lesion_img = combine_lesions(path_raw, lesion_str=data['lesion_str'])
    next_subj['RawLesionSize'] = int(np.sum(lesion_img.get_fdata()))
    next_subj['RawSize_x'], next_subj['RawSize_y'], \
        next_subj['RawSize_z'] = lesion_img.shape

    # 2. remove the skull (from t1 and mask)
    print(f's{next_id}: stripping skull')
    file_in = find_file(path=path_raw,
                        include_str=data['t1_inc_str'],
                        exclude_str=data['t1_exc_str'])
    assert len(file_in) == 1  # only a single T1 file should be found
    t1_file = os.path.join(path_raw, file_in[0])
    t1_no_skull_file = os.path.join(path_results, 't1_no_skull.nii.gz')
    mask_no_skull_file = os.path.join(path_results, 'mask_no_skull.nii.gz')

    no_skull_t1_img, mask_img = strip_skull_mask(
        t1_file, t1_no_skull_file, mask_no_skull_file)
    no_skull_lesion_img = apply_mask_to_image(mask_img, lesion_img)

    no_skull_lesion_file = os.path.join(path_results,
                                        'no_skull_lesion.nii.gz')
    no_skull_lesion_img.to_filename(no_skull_lesion_file)
    assert no_skull_lesion_img.shape == no_skull_t1_img.shape

    # 3. correct bias
    print(f's{next_id}: correcting bias. this might take a while')
    t1_no_skull_file_bias = bias_field_correction(t1_no_skull_file)

    # 4. align the image, normalize to mni space
    print(f's{next_id}: normalizing to mni space')
    no_skull_norm_t1_file = os.path.join(
        path_results, 'no_skull_norm_t1.nii.gz'
    )
    no_skull_norm_lesion_file = os.path.join(
        path_results, 'no_skull_norm_lesion.nii.gz')

    transform_matrix_file = os.path.join(path_results, 'matrix.mat')

    normalize_to_mni(t1_no_skull_file_bias, no_skull_norm_t1_file,
                     template_brain_no_skull, transform_matrix_file)
    normalize_to_transform(no_skull_lesion_file, no_skull_norm_lesion_file,
                           path_template, transform_matrix_file)

    # TODO: any other steps? resampling?

    # 5. Plot the results
    print(f's{next_id}: plotting and saving figs')

    plot_t1(template_brain, title='template',
            fig_dir=path_figs, fig_file='0_template' + ext_fig)
    plot_t1(template_brain_no_skull, title='template, no skull',
            fig_dir=path_figs, fig_file='0_1_template_no_skull' + ext_fig)
    plot_t1(t1_file, title='original',
            fig_dir=path_figs, fig_file='1_original_t1' + ext_fig)
    plot_mask(mask_no_skull_file, title='mask',
              fig_dir=path_figs, fig_file='2_mask_no_skull' + ext_fig)
    plot_t1(t1_no_skull_file, title='original, no skull',
            fig_dir=path_figs, fig_file='3_original_no_skull' + ext_fig)
    plot_t1(t1_no_skull_file_bias, title='original, no skull',
            fig_dir=path_figs,
            fig_file='3_5_original_no_skull_bias' + ext_fig)
    plot_mask(lesion_img, title='lesion',
              fig_dir=path_figs, fig_file='4_lesion' + ext_fig)
    plot_mask(no_skull_lesion_img, title='lesion, mask',
              fig_dir=path_figs,
              fig_file='5_mask_lesion_no_skull' + ext_fig)
    plot_t1(no_skull_norm_t1_file,  title='t1, no skull, norm',
            fig_dir=path_figs, fig_file='6_t1_no_skull_norm' + ext_fig)
    plot_mask(no_skull_norm_lesion_file,  title='lesion, no skull, norm',
              fig_dir=path_figs,
              fig_file='7_lesion_no_skull_norm' + ext_fig)
    plot_overlay(lesion_img, path_bg=t1_file, title='before',
                 fig_dir=path_figs,
                 fig_file='8_before_t1_lesion' + ext_fig)
    plot_overlay(no_skull_norm_lesion_file, path_bg=no_skull_norm_t1_file,
                 title='after', fig_dir=path_figs,
                 fig_file='9_after_t1_lesion' + ext_fig)
    plt.show()
    plt.close('all')

    # save the info in the .csv file
    print(f'saving the info to the {csv_file}')
    no_skull_norm_lesion_img = load_img(no_skull_norm_lesion_file)
    no_skull_norm_lesion_data = no_skull_norm_lesion_img.get_fdata()
    next_subj['NewLesionSize'] = int(np.sum(no_skull_norm_lesion_data))

    no_skull_norm_t1_img = load_img(no_skull_norm_t1_file)
    assert no_skull_norm_t1_img.shape == no_skull_norm_lesion_data.shape

    next_subj['NewSize_x'], next_subj['NewSize_y'], \
        next_subj['NewSize_z'] = no_skull_norm_lesion_data.shape
    return next_subj


if __name__ == "__main__":
    dataset_name = 'dataset_2'  # also dataset_2, TODO: dataset_healthy
    # rerun_all: if set to True, all the preprocessed data saved
    # so far will be removed
    rerun_all = False  # careful !!
    ext_fig = '.png'
    csv_file = 'subject_info.csv'

    # data to be saved in the .csv file
    column_names = ['RawPath', 'ProcessedPath', 'NewID',
                    'RawSize_x', 'RawSize_y', 'RawSize_z',
                    'NewSize_x', 'NewSize_y', 'NewSize_z',
                    'RawLesionSize', 'NewLesionSize']
    results_dir = 'data/preprocessed/'

    # find mni templates at:
    # http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
    # use mni_icbm152_t1_tal_nlin_asym_09c.nii.gz for lower resolution
    # and smaller data size
    # use mni_icbm152_t1_tal_nlin_asym_09b_hires for higher resolution
    # but larger data size
    template_file = 'mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'
    template_brain = os.path.join('../../data/',
                                  'mne_template',
                                  template_file)

    # find all the directories with the 'nii.gz' files
    ext = '.nii.gz'
    data = read_dataset(dataset_name)
    assert data is not None
    raw_dir = data['raw_dir']
    print(f'Wait. I am searching for "{ext}" files in {raw_dir}')
    path_list = find_dirs(raw_dir=raw_dir, ext=ext)
    n_dirs = len(path_list)

    if rerun_all:
        print(f'cleaning up {results_dir}')
        clean_all(results_dir)

    # strip the skull from template brain
    template_brain_no_skull = os.path.join(results_dir, 'template.nii.gz')
    template_mask = os.path.join(results_dir, 'template_mask.nii.gz')
    strip_skull_mask(template_brain, template_brain_no_skull, template_mask,
                     frac=0.5)

    next_id, df_info = init_base(results_dir, column_names=column_names,
                                 file_name=csv_file)
    # remove all the paths from the path_list which are already stored in the
    raw_paths_stored = np.array(df_info['RawPath'])
    path_list = [path for path in path_list if path not in raw_paths_stored]

    print(f'begining to analyze {n_dirs} patient directories')
    dict_result = Parallel(n_jobs=N_JOBS)(
        delayed(preprocess_image)(
            next_id+idx, path_raw, template_brain_no_skull)
        for idx, path_raw in enumerate(path_list)
    )

    df = pd.DataFrame(dict_result,
                      index=range(next_id, len(dict_result)+next_id))
    df.to_csv(os.path.join(results_dir, csv_file), mode='a', header=False)
    print(f'saved results from {len(dict_result)} patient directories')
