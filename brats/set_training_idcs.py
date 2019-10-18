import glob
import os
from unet3d.generator import pickle_dump, pickle_load, split_list #get_validation_split
import tables

# in the original unet code the training and validation indices are set within
# the function for getting validation and training generators. The can be 
# generated as a new set or reused from the previously generated file.
# New function for normalizing the data was introduced before, this function uses
# validation and training sets and therefore it won't work correctly if the
# sets are regenearted after the normalization. Hence, here it is possible to 
# alter the training_ids.pkl and validation_ids.pkl before running train.py
# to reset training and validation set

def main():
    #data_file = os.path.abspath("brats_data.h5")
    #data_file_open = tables.open_file(data_file, "r")
    training_file = os.path.abspath("training_ids.pkl")
    validation_file = os.path.abspath("validation_ids.pkl")
    overwrite = True
    split_same = False

    # get all the available data files
    data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
        print(subject_dir)
        subject_files = list()
        for modality in ["t1"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        data_files.append(tuple(subject_files))
    
    if overwrite:
        sample_list = list(range(len(data_files)))

        if split_same:
            # set training and validation to be the same (don't use it for real, just for
            # testing purposes)
            split = 0.5
        else:
            split = 0.8

        sample_list = list(range(len(data_files)))
        training_list, validation_list = split_list(sample_list, split=split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)

    training_list = pickle_load(training_file)
    validation_list = pickle_load(validation_file)
    print('training:', training_list)
    print('validation:', validation_list)
    print('common:', set(training_list) & set(validation_list))
    print('len training, validation', len(training_list), len(validation_list))
    print('len common:', len(set(training_list) & set(validation_list)))
    return training_list, validation_list

if __name__ == "__main__":
    main()