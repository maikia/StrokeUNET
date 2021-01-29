import numpy as np
import os
import pandas as pd
from preprocess import read_dataset, find_dirs, combine_lesions
from preprocess import get_mean


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


def read_lesion_info(case):
    """
        given a specific format (raw_dir, t1_name, (lesions_names))
        it opens all the lesions and gets out from them the size and the total
        lesion size and returns it
    """
    if len(case[2]) > 1:
        import pdb; pdb.set_trace()
    lesion_size = 0
    for lesion in case[2]:
        ok, lesion_img = combine_lesions(case[0], lesion_str=lesion)
        lesion_size += np.sum(lesion_img.get_fdata())
    x, y, z = lesion_img.shape
    del lesion_img
    return int(lesion_size), x, y, z


if __name__ == "__main__":
    ext = '.nii.gz'
    is_public = True

    if is_public:
        dataset_name = 'dataset_1'
        filename = 'public.csv'
    else:
        dataset_name = 'dataset_2'
        filename = 'private.csv'
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

    data = []
    if not len(file_list):
        print(f'No data files found in {data_info["raw_dir"]}')
    else:
        print('Please wait. I am gathering information')
        n_file = 0
        n_all_files = len(file_list)
        for next_case in file_list:
            n_file += 1
            print(n_file, '/', n_all_files)
            lesion_size, x, y, z = read_lesion_info(next_case)

            t1_path = os.path.join(next_case[0], next_case[1])
            mean_color = format(get_mean(t1_path), '.4f')
            data.append([next_case[0], next_case[1],
                         len(next_case[2]), x, y, z, lesion_size,
                         mean_color])

        pd_data = pd.DataFrame.from_records(data, columns=column_names)
        save_file = os.path.join(path_analysis, filename)
        pd_data.to_csv(save_file)
        print(f'Saved {n_all_files} rows to {save_file}')

