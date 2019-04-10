import os
from unet3d.generator import get_validation_split, pickle_dump
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
    data_file = os.path.abspath("brats_data.h5")
    data_file_open = tables.open_file(data_file, "r")
    training_file = os.path.abspath("training_ids.pkl")
    validation_file = os.path.abspath("validation_ids.pkl")
    overwrite = False
    split_same = False

    
    if split_same and overwrite == True:
        # set training and validation to be the same (don't use it for real, just for
        # testing purposes)
        training_list, validation_list = get_validation_split(data_file_open,
                        data_split=0.5,
                        overwrite=overwrite,
                        training_file=training_file,
                        validation_file=validation_file)
        pickle_dump(training_list, validation_file)
    else:
        # setting random validation split
        training_list, validation_list = get_validation_split(data_file_open,
                        data_split=0.8,
                        overwrite=overwrite,
                        training_file=training_file,
                        validation_file=validation_file)

    print('training:', training_list)
    print('validation:', validation_list)
    print('common:', set(training_list) & set(validation_list))
    print('len training, validation', len(training_list), len(validation_list))
    print('len common:', len(set(training_list) & set(validation_list)))
    return training_list, validation_list


if __name__ == "__main__":
    main()