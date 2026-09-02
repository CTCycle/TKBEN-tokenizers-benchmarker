from __future__ import annotations

from typing import Any

import numpy as np


###############################################################################
def compute_vocabulary_shape_metrics(
    vocabulary_rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute compact vocabulary-shape metrics without materializing extra lists."""
    if not vocabulary_rows:
        return {
            "token_length_std": 0.0,
            "token_length_p90": 0.0,
            "token_length_cv": 0.0,
            "single_character_token_percentage": 0.0,
        }

    lengths = np.fromiter(
        (len(str(row.get("token", ""))) for row in vocabulary_rows),
        dtype=np.float64,
        count=len(vocabulary_rows),
    )
    if lengths.size == 0:
        return {
            "token_length_std": 0.0,
            "token_length_p90": 0.0,
            "token_length_cv": 0.0,
            "single_character_token_percentage": 0.0,
        }

    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    p90_length = float(np.percentile(lengths, 90.0))
    token_length_cv = std_length / mean_length if mean_length > 0.0 else 0.0
    single_character_percentage = float(np.mean(lengths == 1.0) * 100.0)

    return {
        "token_length_std": std_length,
        "token_length_p90": p90_length,
        "token_length_cv": token_length_cv,
        "single_character_token_percentage": single_character_percentage,
    }
