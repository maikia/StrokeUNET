import numpy as np
import os
from preprocess import read_dataset, init_dict, find_dirs, combine_lesions


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


def read_lesion_info(case):
    # given a specific format (raw_dir, t1_name, (lesions_names))
    # it opens all the lesions and gets out from them the size and the total
    # lesion size and returns it
    lesion_size = 0
    for lesion in case[2]:
        ok, lesion_img = combine_lesions(case[0], lesion_str=lesion)
        lesion_size += np.sum(lesion_img.get_fdata())
    x, y, z = lesion_img.shape
    return lesion_size, x, y, z


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
    if not os.path.exists(path_analysis):
        os.mkdir(path_analysis)

    data_info = read_dataset(dataset_name)
    assert data_info is not None
    file_list = list_files(data_info)

    column_names = ['RawPath', 'n_lesions',
                    'RawSize_x', 'RawSize_y', 'RawSize_z',
                    'RawLesionSize', 'AverageGrey']

    for next_case in file_list:
        lesion_size, x, y, z = read_lesion_info(next_case)


    import pdb; pdb.set_trace()
    next_subj = init_dict(column_names, RawPath=path_raw,
                          ProcessedPath=path_results, NewID=next_id)
