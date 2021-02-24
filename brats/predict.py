import os
from train import config
from unet3d.prediction import run_validation_cases

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities='NewT1_name',
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         overlap=config["validation_patch_overlap"])


if __name__ == "__main__":
    main()
