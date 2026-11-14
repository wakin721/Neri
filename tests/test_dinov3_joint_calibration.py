import numpy as np
import pytest

from system.dinov3.joint_calibration import fit_joint_threshold, threshold_frontier
from scripts.compare_dinov3_joint_calibration import cross_validate, make_group_folds


def test_joint_solution_matches_independent_brute_force_with_species_balance():
    scores=np.asarray([-.5, .3, .3, 1.2, .8, -.6, .3, 2.])
    labels=np.asarray(["A", "A", "A", "B", "B", "U", "U", "V"])
    known=np.isin(labels,["A","B"])
    rows=[]
    for threshold in np.r_[scores.min(),np.nextafter(np.unique(scores),np.inf)]:
        rejected=scores<threshold
        frr=np.mean([rejected[labels==c].mean() for c in ("A","B")])
        ur=np.mean([rejected[labels==c].mean() for c in ("U","V")])
        rows.append((.5*frr+.5*(1-ur),threshold,frr,ur))
    expected=min(rows,key=lambda r:(round(r[0],12),r[1]))
    actual=fit_joint_threshold(scores,labels,known)
    assert actual["threshold"]==expected[1]
    assert actual["joint_loss"]==pytest.approx(expected[0])


def test_known_samples_change_selected_cutoff_with_identical_unknowns():
    labels=np.asarray(["A","A","U","U"])
    known=np.asarray([True,True,False,False])
    high=fit_joint_threshold([2,2,0,1],labels,known)
    low=fit_joint_threshold([-2,-2,0,1],labels,known)
    assert high["threshold"]!=low["threshold"]
    assert high["known_acceptance_macro"]==1 and high["unknown_rejection_macro"]==1


def test_identical_known_unknown_scores_report_infeasible_joint_constraints():
    actual=fit_joint_threshold([1,1],["A","U"],[True,False],
                               min_known_acceptance=.8,min_unknown_rejection=.9)
    assert actual["feasible"] is False and actual["threshold"] is None
    assert actual["maximum_known_acceptance_at_unknown_constraint"]==0
    assert actual["maximum_unknown_rejection_at_known_constraint"]==0


def test_unequal_photo_counts_do_not_change_population_or_species_weights():
    actual=fit_joint_threshold([-.3]*100+[2,0],["A"]*100+["B","U"],[True]*101+[False])
    assert actual["known_false_rejection_macro"]==pytest.approx(.5)
    assert actual["unknown_rejection_macro"]==1
    assert actual["joint_loss"]==pytest.approx(.25)


def test_rejection_ties_are_never_split_and_boundaries_match_strict_comparison():
    scores=np.asarray([.2,.2,.5,.5])
    frontier=threshold_frontier(scores,["A","U","A","U"],[True,False,True,False])
    assert len(frontier["thresholds"])==3
    for j,threshold in enumerate(frontier["thresholds"]):
        np.testing.assert_allclose(frontier["known_false_rejection"][j],(scores[[0,2]]<threshold).mean())
        np.testing.assert_allclose(frontier["unknown_rejection"][j],(scores[[1,3]]<threshold).mean())


@pytest.mark.parametrize("scores,labels,known", [([np.nan,1],["A","U"],[True,False]),
    ([1,2],["A","U"],[True,True]),([1,2],["A","A"],[True,False])])
def test_bad_population_or_score_inputs_are_rejected(scores,labels,known):
    with pytest.raises(ValueError):
        fit_joint_threshold(scores,labels,known)


def test_camera_component_folds_never_split_a_group():
    groups=np.repeat(np.arange(5),3)
    known=np.tile([True,True,False],5)
    folds=make_group_folds(groups,known)
    assert len(np.unique(folds))==5
    for group in np.unique(groups):
        assert len(np.unique(folds[groups==group]))==1
    for fold in np.unique(folds):
        assert known[folds==fold].any() and (~known[folds==fold]).any()


def test_validation_fold_known_outlier_cannot_change_its_training_normalizer_or_threshold():
    groups=np.repeat(np.arange(5),3)
    labels=np.tile(["A","A","U"],5)
    data=dict(labels=labels,classes=np.asarray(["A"]),predictions=np.asarray(["A"]*15))
    pooled=np.asarray([v for g in range(5) for v in (2+g*.1,2.1+g*.1,g*.1)])
    radial=np.ones(15)
    before=cross_validate(data,pooled,radial,groups)
    changed=pooled.copy()
    changed[0]=1000000
    after=cross_validate(data,changed,radial,groups)
    np.testing.assert_array_equal(before[0],after[0])
    target=int(before[0][0])
    expected={r["policy"]:r["threshold"] for r in before[4] if r["fold"]==target}
    actual={r["policy"]:r["threshold"] for r in after[4] if r["fold"]==target}
    assert actual==expected
