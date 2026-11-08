"""Map parser output to an oracle answer. Kept free of torch so the UI can
import it without loading the LLM stack."""


def parsed_to_y(parsed, options=("A", "B")) -> tuple[int | None, str]:
    """Map parser output → oracle answer y for a 2-item query.

    Returns (y, status). y=None means skip the belief update; the step still
    counts and (i, j) are still marked used.
    """
    if parsed is None:
        return None, "malformed"
    if parsed == "*":
        return None, "uncertain"
    if parsed == []:
        return None, "reject_all"
    if not isinstance(parsed, list):
        return None, "malformed"
    if len(parsed) == 2:
        return None, "both"
    if len(parsed) == 1 and parsed[0] in options:
        return options.index(parsed[0]), "clean"
    return None, "malformed"
