"""Shaping records into what a tool hands back to its caller."""

import dataclasses
from collections.abc import Collection

from ..common import PaperData


def as_response_dict(paper: PaperData, dropped: Collection[str]) -> dict:
    """`paper` as a plain dict for a tool response, without the `dropped` fields.

    Which fields are noise depends on how the tool found the paper, so each tool names
    its own rather than this helper guessing.
    """
    return {
        key: value
        for key, value in dataclasses.asdict(paper).items()
        if key not in dropped
    }
