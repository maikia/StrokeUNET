import os
import pandas as pd
import sys

sys.path.append('./')
from unet3d.data import open_data_file, write_data_to_file  # noqa: E402
from unet3d.generator import get_training_and_validation_generators  # noqa: E402
from unet3d.model import isensee2017_model  # noqa: E402
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model  # noqa: E402
from unet3d.utils.utils import find_dirs  # noqa: E402

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


config = dict()
config["pool_size"] = (2, 1, 2)
# We will be doing max pooling 3 times, so that the size of the image must be
# divisible by 8 if the pooling size is 2. We are leaving the last dimension to
# its original value because we keep max pooling third dim 1
config["image_shape"] = (200, 240, 189)  # original size: (197, 233, 189)
# will be cropped/resampled to
# (the above dims lead to an error:
# ValueError: A `Concatenate` layer requires inputs with matching shapes
# except for the concat axis. Got in puts shapes:
# [(None, 512, 50, 60, 10), (None, 256, 50, 60, 5)]
config["patch_shape"] = (200, 9, 200)  # switch to None to train on the
# whole image (cannot due to memory errors)
config["labels"] = (1)  # the label numbers on the input image, eg (1, 2, 4)
config["n_base_filters"] = 16
config["n_labels"] = 1  # we only have 0 or 1 for a lesion
config["nb_channels"] = 1
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] +
                                  list(config["patch_shape"])
                                  )
else:
    config["input_shape"] = tuple([config["nb_channels"]] +
                                  list(config["image_shape"])
                                  )
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead
# of deconvolution
config["batch_size"] = 6
config["validation_batch_size"] = 12
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs
# if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs
# without the validation loss improving
config["initial_learning_rate"] = 1e-5
config["learning_rate_drop"] = 0.5  # factor by which the learning rate
# will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for
# training
config["flip"] = False  # augments the data by randomly flipping
# an axis
config["permute"] = False  # data shape must be a cube. Augments the data by
# permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = False
config["validation_patch_overlap"] = 5  # if > 0, during training, validation
# patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the
# first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will
# be skipped
config["data_dir"] = "data/"
config["data_file"] = os.path.abspath("stroke_data.h5")
config["model_file"] = os.path.abspath("unet_model.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")
config["overwrite_data"] = True  # If True, will overwrite previous files.
# If False, will use previously written files.
config["overwrite_model"] = True


def _fetch_training_data_files(data_type='public'):
    # read data files paths from the .csv file
    # data_type might be 'public' or 'private'
    data_dir = config["data_dir"]
    data = pd.read_csv(os.path.join(data_dir,
                                    'data_analysis', data_type + '.csv'))
    data_path = os.path.join(data_dir, data_type)
    data_paths = data[
        ['NewT1_name', 'NewMask_name']].apply(
            lambda s: data_path + '/' + s)
    training_files = data_paths.values.tolist()
    return training_files


def _save_new_h5_datafile(data_file_h5, new_image_shape):
    training_files = _fetch_training_data_files('private')

    # write all the data files into the hdf5 file
    # if necessary crop the data to the new dimensions (if less than original)
    # or add the 0 layer around it (if more than original)
    write_data_to_file(training_files,
                       data_file_h5,
                       image_shape=new_image_shape)


def main(overwrite_data=False, overwrite_model=False):
    # run if the data not already stored hdf5
    if overwrite_data or not os.path.exists(config["data_file"]):
        _save_new_h5_datafile(config["data_file"],
                              new_image_shape=config["image_shape"])

    data_file_opened = open_data_file(config["data_file"])

    if not overwrite_model and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model

        print('initializing new isensee model with input shape',
              config['input_shape'])
        '''
        model = isensee2017_model(
            input_shape=config["input_shape"],
            n_labels=config["n_labels"],
            initial_learning_rate=config["initial_learning_rate"],
            n_base_filters=config["n_base_filters"])
        '''
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"],
                              n_base_filters=config["n_base_filters"])

    # get training and testing generators
    (train_generator, validation_generator, n_train_steps,
     n_validation_steps) = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite_data,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite_data=config["overwrite_data"],
         overwrite_model=config["overwrite_model"])
