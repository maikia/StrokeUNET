import os
from preprocess import read_dataset, init_dict, find_dirs


if __name__ == "__main__":
    ext = '.nii.gz'
    is_public = True

    if use_public:
        dataset_name = 'dataset_1'
    else:
        dataset_name = 'dataset_2'
    path_analysis = '/data/data_analysis'
    if is_public:
        os.

    data_info = read_dataset(dataset_name)
    assert data_info is not None
    raw_dir = data_info['raw_dir']
    print(f'Wait. I am searching for "{ext}" files in {raw_dir}')
    path_list = find_dirs(raw_dir=raw_dir, ext=ext)
    n_dirs = len(path_list)

    column_names = ['RawPath', 'N_lesions',
                    'RawSize_x', 'RawSize_y', 'RawSize_z',
                    'RawLesionSize', 'AverageGrey']
    next_subj = init_dict(column_names, RawPath=path_raw,
                          ProcessedPath=path_results, NewID=next_id)