import numpy as np
import os
import pandas as pd
from preprocess import read_dataset, find_dirs, combine_lesions
from preprocess import get_mean, get_nifti_data, normalize_intensity


def normalize_t1_intensity(t1_path, save_path, new_intensity):
    t1_data = get_nifti_data(t1_path)
    old_mean = get_mean(t1_data)
    t1_img = normalize_intensity(t1_path, new_intensity)
    new_mean = get_mean(t1_img.get_fdata())

    if not np.isclose(new_intensity, new_mean):
        print(f'something wrong, new intensity is {new_mean} instead of '
              f'{new_intensity}')
    if save_path and ('nii.gz' in save_path):
        t1_img.to_filename(save_path)
        print('saved t1 file to ', {save_path})
    del t1_data, t1_img
    return old_mean


def list_files(data_info):
    raw_dir = data_info['raw_dir']
    lesion_str = data_info['lesion_str']
    t1_str = data_info['t1_inc_str']
    path_list = find_dirs(raw_dir=raw_dir, ext=ext)
    file_list = []
    for next_dir in path_list:
        t1_files_here = [f for f in os.listdir(next_dir) if
                         (t1_str in f and lesion_str not in f)]
        for next_t1 in t1_files_here:
            # find lesion(s) corresponding to this T1 file
            t1_id = next_t1.split('_')[0]
            lesion_files_here = [f for f in os.listdir(next_dir) if
                                 (t1_id in f and lesion_str in f)]
            file_list.append((next_dir, next_t1, lesion_files_here))
    return file_list


def find_common_prefix(string_list):
    if len(string_list) == 0:
        # return empty string as no words in this string
        return ''
    elif len(string_list) == 1:
        # return the only word which is in the list
        return string_list[0]
    # check which prefix is common
    prefix = ''
    shortest_len = min([len(word) for word in string_list])
    for idx in range(shortest_len):
        next_letter = string_list[0][idx]
        for word in string_list[1:]:
            if word[idx] != next_letter:
                # letters differ, return current prefix
                return prefix
        prefix = prefix + next_letter
    return prefix


def read_lesion_info(case, save_path=None):
    """
        given a specific format (raw_dir, t1_name, (lesions_names))
        it opens all the lesions and gets out from them the size and the total
        lesion size and returns it
        case: dict []
        save_path: str, path where to save the lesion (must be path to nii.gz
        file). if None the lesion will not be saved
    """
    if len(case[2]) > 1:
        common_prefix = find_common_prefix(case[2])
    else:
        common_prefix = case[2][0]

    ok, lesion_img = combine_lesions(case[0], lesion_str=common_prefix)

    lesion_size = np.sum(lesion_img.get_fdata())
    x, y, z = lesion_img.shape

    if save_path and ('nii.gz' in save_path):
        lesion_img.to_filename(save_path)
        print('saved lesion file to ', {save_path})
    del lesion_img
    return int(lesion_size), x, y, z


if __name__ == "__main__":
    ext = '.nii.gz'
    is_public = False
    save_dir = 'data'
    # if True the new T1 and the combined will be saved in a given path
    save_preprocessed_data = True
    # normalize all T1 to the same intentisty (in private raw data it varies
    # a lot)
    new_mean_intensity = 30

    if is_public:
        dataset_name = 'dataset_1'
        filename = 'public.csv'
        save_dir = os.path.join(save_dir, 'public')
    else:
        dataset_name = 'dataset_2'
        filename = 'private.csv'
        save_dir = os.path.join(save_dir, 'private')
    if save_preprocessed_data:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    path_analysis = 'data/data_analysis'
    path_raw = os.path.join(path_analysis, filename)

    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    data_info = read_dataset(dataset_name)
    assert data_info is not None
    file_list = list_files(data_info)

    column_names = ['RawPath', 'T1_filename', 'n_lesions',
                    'RawSize_x', 'RawSize_y', 'RawSize_z',
                    'RawLesionSize', 'AverageGrey']
    if save_preprocessed_data:
        column_names.extend(['NewPath', 'NewT1_name', 'NewMask_name',
                            'NewAverageGrey'])

    data = []
    idx = 1
    if not len(file_list):
        print(f'No data files found in {data_info["raw_dir"]}')
    else:
        print('Please wait. I am gathering information')
        n_file = 0
        n_all_files = len(file_list)
        for next_case in file_list:
            n_file += 1
            print(n_file, '/', n_all_files)
            t1_path = os.path.join(next_case[0], next_case[1])

            if save_preprocessed_data:
                new_lesion_file_name = str(idx) + '_lesion.nii.gz'
                save_path_lesion = os.path.join(save_dir, new_lesion_file_name)
                lesion_size, x, y, z = read_lesion_info(
                    next_case, save_path=save_path_lesion)

                new_t1_filename = str(idx) + '_T1.nii.gz'
                save_path_t1 = os.path.join(save_dir, new_t1_filename)
                old_mean_intensity = normalize_t1_intensity(
                    t1_path, save_path_t1, new_mean_intensity)
                idx += 1
            else:
                lesion_size, x, y, z = read_lesion_info(next_case)

            if not save_preprocessed_data:
                data.append([next_case[0], next_case[1],
                            len(next_case[2]), x, y, z, lesion_size,
                            old_mean_intensity])
            else:
                data.append([next_case[0], next_case[1],
                             len(next_case[2]), x, y, z, lesion_size,
                             old_mean_intensity, save_dir,
                             new_t1_filename, new_lesion_file_name,
                             new_mean_intensity])

        pd_data = pd.DataFrame.from_records(data, columns=column_names)
        save_file = os.path.join(path_analysis, filename)
        pd_data.to_csv(save_file)
        print(f'Saved {n_all_files} rows to {save_file}')
