"""Species-balanced joint threshold selection for a high-is-known score.

This is an offline selector. It does not change the deployed Macro90 learning
policy. Known rejection and Unknown false acceptance receive separate masses,
regardless of how many photos each population contains.
"""
from __future__ import annotations

import numpy as np

from .balanced_unknown_calibration import unknown_weights


def threshold_frontier(scores, labels, known_mask):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    known = np.asarray(known_mask, dtype=bool)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or labels.shape != scores.shape or known.shape != scores.shape):
        raise ValueError("Finite aligned score, species and known-mask vectors required")
    if not known.any() or known.all():
        raise ValueError("Joint calibration requires both Known and Unknown")
    if set(labels[known]) & set(labels[~known]):
        raise ValueError("Known/Unknown roles must be consistent within each species")
    kw, uw = np.zeros(len(scores)), np.zeros(len(scores))
    kw[known] = unknown_weights(labels[known])
    uw[~known] = unknown_weights(labels[~known])
    order = np.argsort(scores, kind="stable")
    unique, begin = np.unique(scores[order], return_index=True)
    successor = np.nextafter(unique, np.inf)
    if not np.isfinite(successor).all():
        raise ValueError("Scores must allow finite threshold successors")
    # One candidate before any rejection, then one after each entire tie group.
    thresholds = np.r_[unique[0], successor]
    known_rejected = np.clip(np.r_[0., np.cumsum(np.add.reduceat(kw[order], begin))], 0, 1)
    unknown_rejected = np.clip(np.r_[0., np.cumsum(np.add.reduceat(uw[order], begin))], 0, 1)
    return dict(thresholds=thresholds, known_false_rejection=known_rejected,
                known_acceptance=1-known_rejected, unknown_rejection=unknown_rejected,
                unknown_false_acceptance=1-unknown_rejected)


def fit_joint_threshold(scores, labels, known_mask, *, unknown_cost=.5,
                        min_unknown_rejection=None, min_known_acceptance=None):
    """Minimize (1-cost)*macro Known FRR + cost*macro Unknown FAR on C.

    All tied scores cross a threshold together. Among equally good solutions,
    choose the least strict threshold. Infeasible constraints return a witness
    instead of silently relaxing a requested bound.
    """
    if not np.isfinite(unknown_cost) or not 0 < unknown_cost < 1:
        raise ValueError("unknown_cost must be in (0, 1)")
    for value in (min_unknown_rejection, min_known_acceptance):
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Acceptance/rejection constraints must be in [0, 1]")
    frontier = threshold_frontier(scores, labels, known_mask)
    frr = frontier["known_false_rejection"]
    far = frontier["unknown_false_acceptance"]
    kr, ur = frontier["known_acceptance"], frontier["unknown_rejection"]
    satisfy_u = np.ones(len(frr), dtype=bool) if min_unknown_rejection is None else ur >= min_unknown_rejection-1e-12
    satisfy_k = np.ones(len(frr), dtype=bool) if min_known_acceptance is None else kr >= min_known_acceptance-1e-12
    feasible = satisfy_u & satisfy_k
    loss = (1-unknown_cost)*frr+unknown_cost*far
    result = dict(policy="species_balanced_joint_loss", unknown_cost=float(unknown_cost),
                  min_unknown_rejection=min_unknown_rejection, min_known_acceptance=min_known_acceptance,
                  tie_break="minimum joint loss within 1e-12, then least strict threshold",
                  feasible=bool(feasible.any()), candidate_cutoffs=len(loss), threshold=None,
                  maximum_known_acceptance_at_unknown_constraint=float(kr[satisfy_u].max()) if satisfy_u.any() else None,
                  maximum_unknown_rejection_at_known_constraint=float(ur[satisfy_k].max()) if satisfy_k.any() else None)
    if not feasible.any():
        return result
    optimum = float(loss[feasible].min())
    chosen = int(np.flatnonzero(feasible & (loss <= optimum+1e-12))[0])
    result.update(threshold=float(frontier["thresholds"][chosen]), joint_loss=float(loss[chosen]),
                  known_false_rejection_macro=float(frr[chosen]), known_acceptance_macro=float(kr[chosen]),
                  unknown_rejection_macro=float(ur[chosen]), unknown_false_acceptance_macro=float(far[chosen]))
    return result
