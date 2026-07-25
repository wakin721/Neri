from collections.abc import Iterable


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
