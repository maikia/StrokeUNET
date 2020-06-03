"""
Tools for converting, normalizing, and fixing the stroke data.
"""

import glob
import os
import warnings
import shutil

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from brats.train import config


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


def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + ".nii.gz")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def check_origin(in_file, in_file2):
    """ TODO: add doc"""
    image = sitk.ReadImage(in_file)
    image2 = sitk.ReadImage(in_file2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_file)


def normalize_image(in_file, out_file, bias_correction=True):
    """ TODO: for now no normalization is done, add it + add the doc """
    if bias_correction:
        pass
        # correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file


def convert_stroke_data(stroke_folder='../data/ATLAS_R1.1/',#data/raw', #../../../stroke/data/images', 
        out_folder='data/preprocessed'): #, overwrite =True):
    """
    TODO: correct the doc
    Preprocesses the BRATS data and writes it to a given output folder. Assumes the original folder structure.
    :param brats_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    """
    Copies and renames the data files to run within this program settings
    it also changes all the nonzero values in the mask to 1
    """
    # create data/preprocessed directory with files t1 and truth
    print('preparing: ', glob.glob(os.path.join(stroke_folder, "*")))
    for site_folder in glob.glob(os.path.join(stroke_folder, "*")):
        if os.path.isdir(site_folder):
            for subject_folder in glob.glob(os.path.join(site_folder, "*")):
                if os.path.isdir(subject_folder):
                    subject = os.path.basename(subject_folder)
                    new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
                    if not os.path.exists(new_subject_folder):
                        os.makedirs(new_subject_folder)
                    image_file = get_stroke_image(subject_folder)
                    truth_file = get_stroke_image(subject_folder, name='LesionSmooth_stx')

                    out_file_t1 = os.path.abspath(os.path.join(new_subject_folder, "t1.nii.gz"))
                    shutil.copy(image_file, out_file_t1)

                    out_file_path = os.path.abspath(os.path.join(new_subject_folder, "truth.nii.gz"))
                    #shutil.copy(truth_file, out_file_mask) # only copies the mask

                    # set all of the positive values in the mask to 1

                    truth_image = sitk.ReadImage(truth_file)
                    truth_array = sitk.GetArrayFromImage(truth_image)
                    #print(np.unique(truth_array))
                    truth_array[truth_array > 0] = 1
                    truth_mask = sitk.GetImageFromArray(truth_array, isVector=False)
                    sitk.WriteImage(truth_mask, out_file_path)


def convert_healthy_data(healthy_folder='../data/healthy',
    out_folder='data/preprocessed/healthy'):
    """
    TODO: correct the doc
    Preprocesses the BRATS data and writes it to a given output folder. Assumes the original folder structure.
    :param brats_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    """
    Copies and renames the data files to run within this program settings
    for the healthy set of mri images. For each scan it adds the mask of the given
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
    empty_mask = make_empty_mask(dim=[189,233,197]) # TODO: change to variable

    for subject_file in glob.glob(os.path.join(healthy_folder, "*")):
        subject = os.path.basename(subject_file)
        new_subject_dir = os.path.join(out_folder, subject.split('_',2)[1])
        if not os.path.exists(new_subject_dir): 
                        os.makedirs(new_subject_dir)
        out_file_t1 = os.path.abspath(os.path.join(new_subject_dir, 
                        "t1.nii.gz"))

        shutil.copy(subject_file, out_file_t1)

        # paste the mask with no lesion here
        out_file_mask = os.path.abspath(os.path.join(new_subject_dir, 
                        "truth.nii.gz"))
        sitk.WriteImage(empty_mask, out_file_mask)


def make_empty_mask(dim=[189,233,197]):
    array_mask = np.zeros([dim[0],dim[1],dim[2]])
    empty_mask = sitk.GetImageFromArray(array_mask, isVector=False)
    return empty_mask

def get_stroke_image(subject_folder, subfolder='t01', name='t1'):

    file_card = os.path.join(subject_folder, subfolder, 
                    "*" + name + "*.nii.gz")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


if __name__ == "__main__":
    data_dir = ''
    healthy_data_dir = ''
    output_data_dir = ''
