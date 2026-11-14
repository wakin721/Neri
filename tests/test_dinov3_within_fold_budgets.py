import numpy as np
import pytest

from scripts.compare_dinov3_within_fold_budgets import fit_selected_negatives


def data():
    return dict(labels=np.asarray(['A','A','U','U','V','V']),known=np.asarray([True,True,False,False,False,False]),
                known_score=np.asarray([2.,3.,0.,100.,1.,100.]))


def test_unselected_unknown_labels_and_scores_do_not_enter_threshold_fit():
    first=data();second=data();second['known_score'][[3,5]]=100000
    assert fit_selected_negatives(first,[2,4])==fit_selected_negatives(second,[2,4])


def test_known_rows_cannot_be_mislabeled_as_simulated_unknown():
    with pytest.raises(ValueError,match='actual held-out Unknown'):
        fit_selected_negatives(data(),[0,2,4])


def test_budget_must_keep_fixed_unknown_species_coverage():
    with pytest.raises(ValueError,match='species coverage'):
        fit_selected_negatives(data(),[2])
