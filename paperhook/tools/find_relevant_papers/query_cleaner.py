
import re

WILDCARDS = re.compile(r"[*?]")

def remove_wildcards(s: str) -> str:
    """Replace wildcard characters (``*`` and ``?``) with spaces.

    Search APIs often treat ``*`` and ``?`` as wildcards, which breaks
    queries where those characters are just punctuation (e.g. a title
    ending in a question mark). Replacing them with a space rather than
    deleting them avoids joining words that were separated by a wildcard.

    Args:
        s: The raw query string.

    Returns:
        The string with wildcards replaced by spaces and surrounding
        whitespace stripped. Note that interior runs of whitespace are
        left as-is.

    Example:
        >>> remove_wildcards("Is peer review effective?")
        'Is peer review effective'
    """
    return WILDCARDS.sub(" ", s).strip()