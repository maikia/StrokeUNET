import os
import pandas as pd
import numpy as np
from preprocess import find_dirs, read_dataset
from analyze_raw import read_lesion_info, list_files, normalize_t1_intensity


def add_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == "__main__":
    ext = '.nii.gz'
    save_dir = 'data'
    save_dir_public_train = 'data/public/train'
    save_dir_public_test = 'data/public/test'
    save_dir_private_train = 'data/private/train'
    save_dir_private_test = 'data/private/test'
    p_public = 0.6  # from the total of all the files
    p_train = 0.8
    np.random.seed(42)
    filename = 'data_info.csv'

    add_dir(save_dir_public_train)
    add_dir(save_dir_public_test)
    add_dir(save_dir_private_train)
    add_dir(save_dir_private_test)

    atlas_dataset = 'dataset_1'  # already public, old dataset
    new_dataset = 'dataset_4'
    data_info_atlas = read_dataset(atlas_dataset)
    assert data_info_atlas is not None
    data_info_new = read_dataset(new_dataset)
    assert data_info_new is not None
    file_list_atlas = list_files(data_info_atlas)
    file_list_new = list_files(data_info_new)

    n_atlas = len(file_list_atlas)
    n_new = len(file_list_new)
    n_total = n_atlas + n_new

    # calculate the number of samples required in each dataset
    n_public = max(int(0.6 * n_total), n_atlas)
    n_private = n_total - n_public
    n_public_train = int(n_public * p_train)
    n_public_test = n_public - n_public_train
    n_private_train = int(n_private * p_train)
    n_private_test = n_private - n_private_train
    assert (n_public_train + n_public_test + n_private_train + n_private_test
            == n_total)

    # how many private samples we need to switch to public?
    n_private_to_public = n_public - n_atlas
    p_private_to_public = n_private_to_public / n_new

    # read all the info from the datasets and save to the file
    # site
    # lesion size
    # lesion intensity
    # is public
    # assign the number
    numbers = np.arange(n_total)
    np.random.shuffle(numbers)

    column_names = ['RawPath', 'T1_filename', 'site', 'n_lesion_files',
                    'Size_x', 'Size_y', 'Size_z', 'NewPath', 'NewT1_name',
                    'NewMask_name',
                    'LesionSize', 'AverageGrey', 'is_public', 'is_train']
    file_list = file_list_atlas
    file_list.extend(file_list_new)
    is_public = np.zeros(n_total)
    is_public[:n_atlas] = 1
    data = []

    if not len(file_list):
        print(f'No data files found in {data_info["raw_dir"]}')
    else:
        print('Please wait. I am gathering information')
        n_file = 0
        for next_case in file_list:
            n_assigned = numbers[n_file]
            print(n_file + 1, '/', n_total)
            t1_path = os.path.join(next_case[0], next_case[1])
            dirs = t1_path.split('/')
            for d in dirs:
                if 'Site' in d or 'R0' in d:
                    site = d
                    break
                # TODO: should catch if not found

            new_lesion_file_name = str(n_assigned) + '_lesion.nii.gz'
            new_t1_filename = str(n_assigned) + '_T1.nii.gz'

            # randomly select if it is public or private (unless must remain
            # pubic)
            if is_public[n_file]:
                it_is_public = True
            else:
                it_is_public = np.random.choice([True, False],
                                                p=[p_private_to_public,
                                                   1-p_private_to_public])
            is_train = np.random.choice([True, False], p=[p_train, 1-p_train])
            if it_is_public:
                if is_train:
                    save_dir_exact = os.path.join(save_dir, 'public', 'train')
                else:
                    save_dir_exact = os.path.join(save_dir, 'public', 'test')
            elif not it_is_public:
                if is_train:
                    save_dir_exact = os.path.join(save_dir, 'private', 'train')
                else:
                    save_dir_exact = os.path.join(save_dir, 'private', 'test')
            else:
                pass  # TODO: catch not implemented error
            save_path_lesion = os.path.join(save_dir_exact,
                                            new_lesion_file_name)
            lesion_size, x, y, z = read_lesion_info(
                    next_case, save_path=save_path_lesion)
            save_path_t1 = os.path.join(save_dir_exact, new_t1_filename)
            mean_intensity = round(normalize_t1_intensity(
                    t1_path, save_path_t1, new_intensity=None), 4)
            
            data.append([next_case[0], next_case[1], # 'RawPath', 'T1_filename'
                        site, len(next_case[2]),  # 'site', 'n_lesion_files'
                        x, y, z,  # 'Size_x', 'Size_y', 'Size_z'
                        save_dir_exact,  # 'NewPath'
                        new_t1_filename,  # 'NewT1_name'
                        new_lesion_file_name,  # 'NewMask_name'
                        lesion_size,  # 'LesionSize'
                        mean_intensity,  # 'AverageGrey'
                        it_is_public, is_train])  # 'is_public', 'is_train'
            n_file += 1

    pd_data = pd.DataFrame.from_records(data, columns=column_names)
    save_file = os.path.join(save_dir, filename)
    pd_data.to_csv(save_file)
    import pdb; pdb.set_trace()
    print(f'Saved {n_all_files} rows to {save_file}')
    # copy the files with the new names to the correct subdirs
    # make new directories

    # plot the distributions
    # 1. lesion_size
    # 2. lesion_intensity



