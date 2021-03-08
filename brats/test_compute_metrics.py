import numpy as np
import pytest

from compute_metrics import DiceCoeff, HausdorffDistance, Precision, Recall
from compute_metrics import dummy_predict


@pytest.mark.parametrize("Score, min_score",
                         [(DiceCoeff, 1.0),
                          (HausdorffDistance, 0.0),
                          (Precision, 1.0),
                          (Recall, 1.0)])
def test_truth_mask_1s(Score, min_score):
    truth = np.ones([10, 20, 30])
    predict = dummy_predict(truth, 1.0)

    score = Score()
    s = score.score_function(truth, predict)

    assert min_score == s

@pytest.mark.parametrize("Score, max_score",
                         [(DiceCoeff, 0.0),
                          (HausdorffDistance, np.inf),
                          (Precision, 0.0),
                          (Recall, 0.0)])
def test_truth_mask_1s(Score, max_score):
    truth = np.ones([10, 20, 30])
    predict = dummy_predict(truth, 0.0)

    score = Score()
    s = score.score_function(truth, predict)

    assert max_score == s
