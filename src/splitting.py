import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def actor_disjoint_split(actor_ids, random_state=42):
    """
    Creates train, validation, and test splits with no shared actors.

    This prevents the model from learning a speaker's voice in training
    and then being evaluated on the same speaker in testing.
    """
    actor_ids = np.asarray(actor_ids)
    indices = np.arange(len(actor_ids))

    # Reserve approximately 20% of actors as a final untouched test set.
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(
        outer_split.split(indices, groups=actor_ids)
    )

    # Split the remaining actors into training and validation sets.
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state + 1,
    )
    train_relative, val_relative = next(
        inner_split.split(
            train_val_idx,
            groups=actor_ids[train_val_idx],
        )
    )

    return (
        train_val_idx[train_relative],
        train_val_idx[val_relative],
        test_idx,
    )