from nilearn.image import load_img
import numpy as np
from skimage import metrics
from sklearn.metrics import precision_score, recall_score


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


def check_mask(mask):
    ''' assert that the given mask consists only of 0s and 1s '''
    assert np.all(np.isin(mask, [0, 1])), ('Cannot compute the score.'
                                           'Found values other than 0s and 1s')


# define the scores
class DiceCoeff(BaseScoreType):
    # Dice’s coefficient (DC), which describes the volume overlap between two
    # segmentations and is sensitive to the lesion size;
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='dice coeff', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = self._dice_coeff(y_true_mask, y_pred_mask)
        return score

    def _dice_coeff(self, y_true_mask, y_pred_mask):
        if (not np.any(y_pred_mask)) & (not np.any(y_true_mask)):
            # if there is no true mask in the truth and prediction
            return 1
        else:
            dice = (
                np.sum(np.logical_and(y_pred_mask, y_true_mask) * 2.0) /
                (np.sum(y_pred_mask) + np.sum(y_true_mask))
                )
        return dice


class Precision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='precision', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        if np.sum(y_pred_mask) == 0 and not np.sum(y_true_mask) == 0:
            return 0.0
        score = precision_score(y_true_mask.ravel(), y_pred_mask.ravel())
        return score


class Recall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='recall', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = recall_score(y_true_mask.ravel(), y_pred_mask.ravel())
        return score


class HausdorffDistance(BaseScoreType):
    # recommened to use 95% percentile Hausdorff Distance which tolerates small
    # otliers
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='Hausdorff', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = metrics.hausdorff_distance(y_true_mask, y_pred_mask)
        return score


class AbsoluteVolumeDifference(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='AVD', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_mask, y_pred_mask):
        check_mask(y_true_mask)
        check_mask(y_pred_mask)
        score = np.abs(np.mean(y_true_mask) - np.mean(y_pred_mask))

        return score


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

    match_prob = 0.0
    pred_arr = dummy_predict(truth_arr, match_prob)

    score_dice = DiceCoeff()
    print('Dice coeff is: ', score_dice.score_function(truth_arr, pred_arr))

    score_hausdorff = HausdorffDistance()
    print('Hausdorff distance is: ',
          score_hausdorff.score_function(truth_arr, pred_arr))

    score_recall = Recall()
    print('Recall is: ', score_recall.score_function(truth_arr, pred_arr))

    score_precision = Precision()
    print('Precision is: ', score_precision.score_function(
        truth_arr, pred_arr))

    score_avd = AbsoluteVolumeDifference()
    print('AVD is: ', score_avd.score_function(
        truth_arr, pred_arr))
