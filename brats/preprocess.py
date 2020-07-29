"""
Tools for converting, normalizing, and fixing the stroke data.
"""

import glob
import os
import warnings
import shutil


import matplotlib.pylab as plt
from nilearn.plotting import plot_anat
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection
import SimpleITK as sitk


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
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


def apply_mask(mask, file_in, file_out):
    from nilearn.masking import apply_mask
    masked_data = apply_mask(file_in, mask)


def strip_skull_mask(file_in, file_out, plot=False):
    # mask param does not seem to do what it's suppose to
    skullstrip = BET(in_file=file_in,
                 out_file=file_out,
                 mask=True)
    res = skullstrip.run()

    # it sets all the values > 0 to 1 creating a mask
    t_img = load_img(file_out)
    mask = math_img('img > 0', img=t_img)
    mask.to_filename(file_out)

    if plot:

        plot_anat(file_in,
              title='original', display_mode='ortho', dim=-1, draw_cross=False,
              annotate=False);
        plot_anat(file_out,
              title='mask2', display_mode='ortho', dim=-1, draw_cross=False,
              annotate=False);
        plt.show()

def strip_skull(file_in, file_out, plot=False)
    ''' file_in and file_out must be of '.nii.gz' format. Need to have fsl
    installed to run'''
    skullstrip = BET()
    skullstrip.inputs.in_file = file_in
    skullstrip.inputs.out_file = file_out
    res = skullstrip.run()


if __name__ == "__main__":
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
