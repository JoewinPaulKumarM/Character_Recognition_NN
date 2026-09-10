"""
Dataset splitting utilities for train / validation / test sets.

Default ratio: 70 % train, 15 % validation, 15 % test.
Splitting is **stratified** — each class keeps roughly the same proportion
across all three sets — and reproducible via an optional random seed.
"""

import numpy as np


def train_val_test_split(
    images: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int | None = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Split images and labels into train, validation, and test sets.

    The split is **stratified**: for every unique label the samples are
    shuffled independently and divided according to the requested ratios,
    so each set mirrors the overall class distribution.

    Parameters
    ----------
    images : np.ndarray, shape (N, ...)
        Image data — any shape whose first axis is the sample axis.
    labels : np.ndarray, shape (N,) or (N, C)
        If 1-D, treated as integer class labels.
        If 2-D (one-hot encoded), the argmax along axis 1 is used for
        stratification, and the original one-hot array is split.
    train_ratio : float
        Fraction of data for training (default 0.70).
    val_ratio : float
        Fraction of data for validation (default 0.15).
    test_ratio : float
        Fraction of data for testing (default 0.15).
    seed : int or None
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to a
    dict with ``"images"`` and ``"labels"`` arrays.

    Raises
    ------
    ValueError
        If the ratios do not sum to 1.0 (within floating-point tolerance)
        or if images and labels have mismatched sample counts.
    """
    # ── Validate inputs ──────────────────────────────────────────────
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio:.4f}"
        )

    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images and labels must have the same number of samples, "
            f"got {images.shape[0]} vs {labels.shape[0]}"
        )

    # ── Determine integer class ids for stratification ───────────────
    if labels.ndim == 2:
        # One-hot encoded → use argmax as the stratification key
        class_ids = np.argmax(labels, axis=1)
    else:
        class_ids = labels

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(class_ids)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for cls in unique_classes:
        cls_indices = np.where(class_ids == cls)[0]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Remainder goes to test to guarantee no sample is lost
        # n_test = n - n_train - n_val

        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train : n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val :])

    # Shuffle within each set so batches are not class-sorted
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    result = {
        "train": {"images": images[train_idx], "labels": labels[train_idx]},
        "val":   {"images": images[val_idx],   "labels": labels[val_idx]},
        "test":  {"images": images[test_idx],  "labels": labels[test_idx]},
    }

    # ── Summary ──────────────────────────────────────────────────────
    total = len(images)
    print(f"[split] Total samples : {total}")
    print(
        f"[split] Train        : {len(train_idx):>7,}  "
        f"({len(train_idx) / total * 100:.1f}%)"
    )
    print(
        f"[split] Validation   : {len(val_idx):>7,}  "
        f"({len(val_idx) / total * 100:.1f}%)"
    )
    print(
        f"[split] Test         : {len(test_idx):>7,}  "
        f"({len(test_idx) / total * 100:.1f}%)"
    )

    return result


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from loader import load_data

    data = load_data(flatten=True, encode_labels=False)
    splits = train_val_test_split(data["images"], data["labels"])

    for name, subset in splits.items():
        print(f"\n{name:>10s}  images {subset['images'].shape}  "
              f"labels {subset['labels'].shape}")
