import glob
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd

matplotlib.use('agg')


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    if np.sum(truth) == 0:
        # if true mask should be 0 and is 0, dice will be 1
        if np.sum(prediction) == 0:
            return 1.
        else:
            return 0
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def main():
    header = ("DiceCoeff", "TruthSize", "PredictSize")

    rows = list()
    subject_ids = list()
    for case_folder in glob.glob("prediction/*"):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        truth[truth > 0] = 1
        rows.append([dice_coefficient(get_whole_tumor_mask(truth),
                    get_whole_tumor_mask(prediction)),
                    np.sum(truth), np.sum(prediction)])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv("./prediction/brats_scores.csv")

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[not np.isnan(values)]
    plt.figure()
    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot.png")
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')
        plt.figure()
        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')


if __name__ == "__main__":
    main()
