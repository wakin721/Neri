from collections.abc import Iterable


def is_dinov3_decision(box: dict) -> bool:
    keys = {'box_known_score', 'sequence_known_score', 'within_score', 'ridge_score', 'raw_ridge_score'}
    candidates = box.get('候选项') or box.get('candidates') or []
    return (any(box.get(key) is not None for key in keys)
            or str(box.get('observation_id', '')).startswith('dinov3:')
            or any(isinstance(candidate, dict) and (
                candidate.get('source') == 'dinov3_ridge_within'
                or any(candidate.get(key) is not None for key in keys)) for candidate in candidates))


def dinov3_final_species(box: dict) -> str:
    """Preserve calibrated model decisions and explicit human overrides."""
    species = str(box.get('物种') or box.get('species') or '').strip()
    if box.get('feedback_status') in {'confirmed', 'image_label'}:
        return species
    candidates = box.get('候选项') or box.get('candidates') or []
    primary = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    if any(source.get(key) is False for source in (box, primary)
           for key in ('accepted', 'box_accepted', 'sequence_accepted')):
        return species if species in {'Unknown', '拒识/Unknown'} else 'Unknown'
    return species or str(primary.get('name') or '').strip()


def combined_confidence_weights(priority: str) -> tuple[float, float]:
    """Return detection/classification weights for a combined inference."""

    if priority == "detection":
        return 0.6, 0.4
    return 0.4, 0.6


def candidate_matches_selected_species(
    raw_name: str,
    translated_name: str,
    selected_species_names: Iterable[str] | None,
) -> bool:
    """Match a classifier candidate against raw, translated, or display names."""

    selected = {
        str(name).strip().casefold()
        for name in (selected_species_names or [])
        if str(name).strip()
    }
    if not selected:
        return True
    aliases = {
        raw_name.strip().casefold(),
        translated_name.strip().casefold(),
    }
    if raw_name != translated_name:
        aliases.add(f"{translated_name} ({raw_name})".strip().casefold())
    return bool(aliases & selected)
