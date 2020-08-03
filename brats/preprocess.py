"""
Tools for converting, normalizing, and fixing the stroke data.
"""

from nipype.interfaces.fsl import BET
import glob
import os
import pandas as pd
import warnings
import shutil

from nilearn.image import load_img, math_img
import matplotlib.pylab as plt
from nilearn.plotting import plot_anat
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection
# import SimpleITK as sitk
from nilearn.image import new_img_like
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
import subprocess


def correct_bias(in_file, out_file, image_type):  # =sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will
    then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be "
                                     "found. Will try using SimpleITK for "
                                     "bias field correction which will take "
                                     "much longer. To fix this problem, add "
                                     "N4BiasFieldCorrection to your PATH "
                                     "system variable. (example: "
                                     "EXPORT PATH=${PATH}:/path/to/ants/bin)"
                                     ))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def rescale(in_file, out_file, minimum=0, maximum=20000):
    image = sitk.ReadImage(in_file)
    sitk.WriteImage(sitk.RescaleIntensity(image, minimum, maximum), out_file)
    return os.path.abspath(out_file)


def normalize_image(in_file, out_file, bias_correction=True):
    """ TODO: for now no normalization is done, add it + add the doc """
    if bias_correction:
        pass
        # correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file


def convert_stroke_data(file_dir, out_dir, scan_name='t1',
                        mask_name='LesionSmooth'):
    """
    Copies and renames the data files to run within this program settings
    it also changes all the nonzero values in the mask to 1

    file_dir: where the files for the single subjects are stored
    out_dir: where the files should be copied to
    scan_name: name which is included in the name of the mri scan
    mask_name: name which is included in the name of the lesion image(s)

    Note1: there might be more than one mask stored for one patients. Those
    masks will be collapsed into one
    Note2: some patients have scans at multiple times. Those scans will be
    considered as separate patient
    the output image masks will always consist only of [0, 1]s

    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # copy the true image
    out_file_t1 = os.path.abspath(os.path.join(out_dir, "t1.nii.gz"))
    mri_file = get_files(file_dir, scan_name, ext='nii.gz')

    # mri_image = sitk.ReadImage(out_file_t1)
    # mri_array = sitk.GetArrayFromImage(mri_image)
    assert len(mri_file) == 1

    shutil.copy(mri_file[0], out_file_t1)

    # mask
    truth_files = get_files(file_dir, name=mask_name)
    assert len(truth_files) >= 1
    out_file_path = os.path.abspath(os.path.join(out_dir, "truth.nii.gz"))
    # if multiple lesions, sum all into one mask
    masks = []
    for mask in truth_files:
        truth_image = sitk.ReadImage(mask)
        truth_array = sitk.GetArrayFromImage(truth_image)
        masks.append(truth_array)
    truth_array = sum(masks)
    # set all of the positive values in the mask to 1
    truth_array[truth_array > 0] = 1
    assert len(np.unique(truth_array)) in [1, 2]

    truth_mask = sitk.GetImageFromArray(truth_array, isVector=False)
    sitk.WriteImage(truth_mask, out_file_path)


def convert_data_new(stroke_folder='../data/BIDS_lesions_zip/',
                     out_folder='data/preprocessed'):
    """
    Finds the subjects data and writes it to a given output folder in a
    consistent format.
    :param stroke_folder: folder containing the data with the following format:
                          |-stroke_folder
                            |_ <subject_name1>
                                |_<subject_name1>nii.gz
                                |_<subject_name1>.gz
    Note: there might be multiple folders at each level.
    """
    # create data/preprocessed directory with files t1 and truth
    subj_dirs = glob.glob(os.path.join(stroke_folder, "*"))
    subj_dirs = [subj_dir for subj_dir in subj_dirs if
                 os.path.isdir(subj_dir)]
    subjects = 0

    print('found {} subject directories'.format(len(subj_dirs)))

    for subj_dir in subj_dirs:
        # subject name
        subject = os.path.basename(subj_dir)
        subjects += 1

        dir_name = 'new_' + subject
        new_subj_dir = os.path.join(out_folder, dir_name)

        convert_stroke_data(file_dir=subj_dir, out_dir=new_subj_dir,
                            scan_name='T1w.', mask_name='label-lesion')
    print('Copied data for {} subjects'.format(subjects))


def find_scan_dirs(raw_dir='../../data/ATLAS_R1.1/'):
    """
    searches for the directories within the given directory and up the tree
    with the scan data (ending on '.nii.gz').
    :raw_dir: directory where to search for the t1 scans
    output: list of directories with scans
    """
    print(f'Wait. I am searching for "nii.gz" files in {raw_dir}')
    path_list = []
    for dirname, dirnames, filenames in os.walk(raw_dir):
        # save directories with all filenames ending on '.nii.gz'
        for filename in filenames:
            if filename[-7:] == '.nii.gz':
                path_list.append(dirname)
                break
    return path_list


def convert_data_old(stroke_folder='../data/ATLAS_R1.1/',
                     out_folder='data/preprocessed'):
    """
    Finds the subjects data and writes it to a given output folder in a
    consistent format.
    :param stroke_folder: folder containing the data with the following format:
                          |-stroke_folder
                            |_ Site1
                                |_ <subject_name1>
                                    |_t01
                                        |_<subject_name1>_t1....nii.gz
                                        |_<subject_name1>_LesionSmooth...nii.gz
    Note: there might be multiple folders at each level.
    """
    # create data/preprocessed directory with files t1 and truth
    site_dirs = glob.glob(os.path.join(stroke_folder, "*"))
    site_dirs = [site_dir for site_dir in site_dirs if os.path.isdir(site_dir)]
    files, subjects = 0, 0
    print('preparing: ', site_dirs)
    print('found {} site directories'.format(len(site_dirs)))
    for site_dir in site_dirs:
        subj_dirs = glob.glob(os.path.join(site_dir, "*"))
        subj_dirs = [subj_dir for subj_dir in subj_dirs if
                     os.path.isdir(subj_dir)]
        site = site_dir.split('/')[-1]
        print('found {} subject directories in {}'.format(len(subj_dirs),
                                                          site))
        for subj_dir in subj_dirs:
            # subject name
            subject = os.path.basename(subj_dir)
            import pdb; pdb.set_trace()

            time_dirs = glob.glob(os.path.join(subj_dir, "*"))
            time_dirs = [time_dir for time_dir in time_dirs if
                         os.path.isdir(time_dir)]
            subjects += 1

            for time_dir in time_dirs:
                dir_name = 's' + site[-1] + '_' + subject + '_t' + time_dir[-1]
                new_subj_dir = os.path.join(out_folder, dir_name)

                convert_stroke_data(file_dir=time_dir, out_dir=new_subj_dir)
                files += 1
    print('Copied data for {} subjects, {} scans'.format(subjects, files))


def convert_healthy_data(healthy_folder='../data/healthy',
                         out_folder='data/preprocessed/healthy'):
    """
    TODO: correct the doc
    Preprocesses the BRATS data and writes it to a given output folder. Assumes
     the original folder structure.
    :param brats_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be
    written
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could
    reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by
    including the given modality in a list
    or tuple.
    :return:
    """
    """
    Copies and renames the data files to run within this program settings
    for the healthy set of mri images. For each scan it adds the mask of the
    given
    size filled with 0s to signify no lesion
    It is assumed that the healthy patient files are named
    'sub_>number_t1_final<.nii.gz'
    """
    print('preparing: ', glob.glob(os.path.join(healthy_folder, "*")))
    # copy the file + filename
    # create out folder if does not already exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # create empty mask, same for all healthy patients
    # TODO: change dim to variable:
    empty_mask = make_empty_mask(dim=[189, 233, 197])
    for subject_file in glob.glob(os.path.join(healthy_folder, "*")):
        subject = os.path.basename(subject_file)
        new_subject_dir = os.path.join(out_folder, subject.split('_', 2)[1])
        if not os.path.exists(new_subject_dir):
            os.makedirs(new_subject_dir)
        out_file_t1 = os.path.abspath(os.path.join(new_subject_dir,
                                                   "t1.nii.gz"))

        shutil.copy(subject_file, out_file_t1)

        # paste the mask with no lesion here
        out_file_mask = os.path.abspath(os.path.join(new_subject_dir,
                                                     "truth.nii.gz"))
        sitk.WriteImage(empty_mask, out_file_mask)


def make_empty_mask(dim=[189, 233, 197]):
    array_mask = np.zeros([dim[0], dim[1], dim[2]])
    empty_mask = sitk.GetImageFromArray(array_mask, isVector=False)
    return empty_mask


def get_files(directory, name='t1', ext='.nii.gz'):
    # In the directory, it finds all the files including the given name and
    # with the givent extension ext

    file_card = os.path.join(directory, "*" + name + "*" + ext)
    try:
        return glob.glob(file_card)
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def apply_mask_to_image(mask, img, file_out, savefig_dir=None, ext='.png'):
    # mask: mask to be applied to the image
    # img, nilearn image to which mask is to be applied to
    # masked: masked image in nilearn format
    # t_img = load_img(mask)
    assert mask.shape == img.shape
    img_data = img.get_fdata()
    img_data[mask == 0] = 0
    masked = new_img_like(img, img_data, affine=None, copy_header=False)

    if savefig_dir is not None:
        plot_anat(masked,
                title='mask', display_mode='ortho', dim=-1, draw_cross=False,
                annotate=False)
        plt.savefig(os.path.join(savefig_dir, '4_mask_lesion_no_skull' + ext))
    return masked


def strip_skull_mask(file_in, file_out, savefig_dir=None, ext='.png'):
    # mask param does not seem to do what it's suppose to
    # and returns the mask
    # set sevefig_dir to None to not plot
    skullstrip = BET(in_file=file_in,
                 out_file=file_out,
                 mask=True)
    res = skullstrip.run()

    if savefig_dir is not None:
        plot_anat(file_in,
              title='original', display_mode='ortho', dim=-1, draw_cross=False,
              annotate=False)
        plt.savefig(os.path.join(savefig_dir, '1_original_image' + ext))
        plot_anat(file_out,
              title='original, no skull', display_mode='ortho',
              dim=-1, draw_cross=False, annotate=False)
        plt.savefig(os.path.join(savefig_dir, '3_original_no_skull' + ext))

    # it sets all the values > 0 to 1 creating a mask
    t_img = load_img(file_out)
    mask = math_img('img > 0', img=t_img)
    mask.to_filename(file_out)

    if savefig_dir:
        plot_anat(file_out,
              title='mask', display_mode='ortho', dim=-1, draw_cross=False,
              annotate=False)
        plt.savefig(os.path.join(savefig_dir, '2_mask_no_skull' + ext))
    return t_img, mask


def combine_lesions(path, lesion_str='Lesion'):
    # path: path to the directory with all the scans
    # out_file: name of the combined lesion file saved in the path
    # lesion_str: string which is included in the name of all the lesions but
    # not any other files
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
        # masked.to_filename(os.path.join(path, out_file))
        return masked
    else:
        # there are no lesions found. This path should be removed
        warnings.warn('there are no lesion files with name including '
                      f'{lesion_str} found in the {path}. This path will be '
                      'removed from further analysis')
        return 0


def strip_skull(file_in, file_out, plot=False):
    ''' file_in and file_out must be of '.nii.gz' format. Need to have fsl
    installed to run'''
    skullstrip = BET()
    skullstrip.inputs.in_file = file_in
    skullstrip.inputs.out_file = file_out
    res = skullstrip.run()


def find_file(path, include='t1', exclude='lesion'):
    files = os.listdir(path)
    if include is not None:
        files = [n_file for n_file in files if (include in n_file)]
    if exclude is not None:
        files = [n_file for n_file in files if (exclude not in n_file)]
    return files


def clean_all(dir_to_clean):
    # removes all the files and directories from the given path

    if os.path.exists(dir_to_clean):
        shutil.rmtree(dir_to_clean)
    os.mkdir(dir_to_clean)
    print(f'cleaned all from {dir_to_clean}')


def init_base(base_dir, file_name = 'subject_info.csv'):
    # initiates filename .csv file if it does not already exists
    # returns next subject ID to be used
    file_path = os.path.join(base_dir, file_name)
    if not os.path.exists(file_path):
        dfObj = pd.DataFrame(columns=['RawPath', 'ProcessedPath',
                                      'NewID', 'RawSize', 'NewSize',
                                      'RawLesionSize', 'NewLesionSize'])
        dfObj.to_csv(file_path)
        return 1, dfObj
    else:
        dfObj = pd.read_csv(file_path)
        if len(dfObj) == 0:
            return 1, dfObj
        else:
            last_used_id = np.max(dfObj['NewID'])
            return last_used_id + 1, dfObj


def init_next_subj(**kwargs):
    # initiates new dictionary with the values to save:
    next_subj = dict.fromkeys(['RawPath', 'ProcessedPath', 'NewID',
                               'RawSize', 'NewSize', 'RawLesionSize',
                               'NewLesionSize'], None)
    for key, value in kwargs.items():
        next_subj[key] = value
    return next_subj


def normalize_to_mni(file_in, file_out, template_out):

    returncode = subprocess.run([
        "flirt",
        "-in", file_in,
        "-out", file_out,
        "-ref", template_out])



    plotting.plot_stat_map(file_in,
                       bg_img=template,
                       cut_coords=(36, -27, 66),
                       threshold=3,
                       title="t-map in original resolution")
    plotting.plot_stat_map(mask,
                        bg_img=template,
                        cut_coords=(36, -27, 66),
                        threshold=3,
                        title="Original mask")
    plotting.plot_stat_map(resampled_stat_img,
                        bg_img=template,
                        cut_coords=(36, -27, 66),
                        threshold=3,
                        title="Resampled t-map")
    plotting.plot_stat_map(resampled_mask,
                        bg_img=template,
                        cut_coords=(36, -27, 66),
                        threshold=3,
                        title="Resampled mask")
    plotting.show()


if __name__ == "__main__":
    # loop through available images
    #   unify all the available masks to a single mask with only 1s and 0s
    #   remove the skull (from t1 and mask)
    #   move all the images (t1 and masks) to mni space
    #   noramlize images (same average color) ??
    raw_dir = '../../data/ATLAS_R1.1/'  # first data set
    raw_dir2 = '../../data/BIDS_lesions_zip/'  # second data set
    raw_dir_healthy = '../../data/healthy'
    results_dir = 'data/preprocessed/'
    ext = '.png'
    # careful, if set to True, all the previous preprocessing saved
    # data will be removed
    rerun_all = True  # careful !!

    # save the template mni brain locally
    template = load_mni152_template()
    template_out = '../../data/template_mni.nii.gz'
    template.to_filename(template_out)

    path_list = find_scan_dirs(raw_dir)
    len_path_list = len(path_list)

    if rerun_all:
        clean_all(results_dir)
    next_id, df_info = init_base(results_dir)

    for idx, path_raw in enumerate(path_list):
        print(f'{idx+1}/{len_path_list}, subject {next_id}, working on {path_raw}')
        path_results = os.path.join(results_dir, f'subject_{next_id}')
        path_figs = os.path.join(path_results, 'figs')

        # create output path (TODO: for now if sth goes wrong this path remains
        os.mkdir(path_results)
        os.mkdir(path_figs)
        # initiates info dict for the new subject
        next_subj = init_next_subj(RawPath=path_raw,
                                   ProcessedPath=path_results,
                                   NewID=next_id)

        # check if multiple lesion files are saved
        # combines them and sets to 0 or 1
        lesion_img = combine_lesions(path_raw, lesion_str='Lesion')
        next_subj['RawLesionSize'] = int(np.sum(lesion_img.get_fdata()))
        next_subj['RawSize'] = lesion_img.shape
        # assert not mask_img
        # remove the skull (from t1 and mask)
        print('stripping skull')
        file_in = find_file(path=path_raw, include='t1', exclude=None)
        assert len(file_in) == 1  # only a single T1 file should be found
        file_in = os.path.join(path_raw, file_in[0])
        file_out = os.path.join(path_results, 'no_skull_mask.nii.gz')
        no_skull_t1_img, mask_img = strip_skull_mask(
            file_in, file_out, savefig_dir=results_dir, ext=ext)
        no_skull_lesion_img = apply_mask_to_image(mask_img, lesion_img,
                                              file_out=None,
                                              savefig_dir=results_dir, ext=ext)
        assert no_skull_lesion_img.shape == no_skull_t1_img.shape

        # align the image
        print('normalizing the images')
        normalize_to_mni(file_in, file_out, template_out)

        import pdb; pdb.set_trace()
        next_id += next_id

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
