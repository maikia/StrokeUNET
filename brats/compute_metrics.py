from nilearn.image import load_img
import numpy as np


class BaseScoreType(object):
    # based on Base score from ramp workflow
    def check_y_pred_dimensions(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should have {} instances, '
                'instead it has {} instances'.format(len(y_true), len(y_pred)))

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_true = ground_truths  # ground_truths.y_pred[valid_indexes]
        y_pred = predictions  # predictions.y_pred[valid_indexes]
        self.check_y_pred_dimensions(y_true, y_pred)
        return self.__call__(y_true, y_pred)


# define the scores
class DiceCoeff(BaseScoreType):
    # Dice’s coefficient (DC), which describes the volume overlap between two
    # segmentations and is sensitive to the lesion size;
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='dice coeff', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        score = self._dice_coeff(y_true_mask, y_pred_mask)
        return score

    def _dice_coeff(self, y_true_mask, y_pred_mask):

        if (np.sum(y_pred_mask) == 0) & (np.sum(y_true_mask) == 0):
            return 1
        else:
            dice = (np.sum(
                (y_pred_mask == 1) & (y_true_mask == 1)
                ) * 2.0) / (np.sum(y_pred_mask) + np.sum(y_true_mask))
        return dice


def dummy_predict(truth, prob):
    # prob is a probability that 1 will still be 1 in a mask
    # and 0 will still be 0 in a mask
    if prob == 1:
        return truth
    elif prob == 0:
        return np.zeros(truth.shape)
    else:
        pred = truth.copy()
        mask_1 = truth == 1
        mask_0 = truth == 0

        pred[mask_1] = np.random.choice(
            [0, 1], np.sum(mask_1), p=[1-prob, prob]
            )
        pred[mask_0] = np.random.choice(
            [0, 1], np.sum(mask_0), p=[prob, 1-prob]
            )
        return pred


if __name__ == "__main__":
    truth = 'data/private/1_lesion.nii.gz'
    truth_arr = load_img(truth).get_fdata()

    match_prob = 0.99
    pred_arr = dummy_predict(truth_arr, match_prob)

    score = DiceCoeff()
    print('Dice coeff is: ', score.score_function(truth_arr, pred_arr))
