"""PAD prediction pipeline — cohort, features, training, calibration, explanation.

lightgbm is imported here, before anything else in the package can pull in
scikit-learn. Importing it *after* scikit-learn crashes it on Windows with an
access violation, because the two ship clashing OpenMP runtimes. Putting it in
the package __init__ means any `pad.*` import is safe regardless of which
submodule the caller reaches for first.
"""

import lightgbm as lgb  # noqa: F401  (import order matters; see docstring)

__version__ = "2.2.0"
