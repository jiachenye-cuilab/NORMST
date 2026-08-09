"""Joint multi-gene masked-spot samples for one standard Visium slice."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from datasets.masked_visium import MaskedVisiumData


class JointMaskedVisiumDataset(Dataset):
    """Treat one spatial mask, rather than one gene, as a training sample.

    Every target spot is hidden across *all* genes. This is essential: leaving
    non-target genes visible at a target spot would leak its local cell-state
    information into the joint predictor.
    """

    def __init__(
        self,
        data: MaskedVisiumData,
        split: str,
        masks_per_epoch: int = 64,
        train_target_fraction: float = 0.25,
        idw_neighbors: int = 6,
        seed: int = 2026,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if masks_per_epoch < 1:
            raise ValueError("masks_per_epoch must be positive")
        if not 0.0 < train_target_fraction < 1.0:
            raise ValueError("train_target_fraction must be between 0 and 1")
        if idw_neighbors < 1:
            raise ValueError("idw_neighbors must be positive")

        self.data = data
        self.split = split
        self.masks_per_epoch = masks_per_epoch
        self.train_target_fraction = train_target_fraction
        self.seed = seed
        self.epoch = 0
        self.height, self.width = data.shape
        self.n_genes = len(data.genes)

        expression_grid = np.zeros(
            (self.n_genes, self.height, self.width), dtype=np.float32
        )
        expression_grid[:, data.spot_rows, data.spot_cols] = data.expression.T
        self.expression_grid = expression_grid

        self.eval_targets = None
        self.eval_baseline = None
        if split != "train":
            self.eval_targets = (
                data.validation_spots if split == "val" else data.test_spots
            )
            neighbors = min(idw_neighbors, len(data.observed_spots))
            tree = cKDTree(data.physical_xy[data.observed_spots])
            distances, indices = tree.query(
                data.physical_xy[self.eval_targets], k=neighbors
            )
            if neighbors == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            weights = 1.0 / np.maximum(distances, 1e-6) ** 2
            weights /= weights.sum(axis=1, keepdims=True)
            neighbor_spots = data.observed_spots[indices]
            neighbor_expression = data.expression[neighbor_spots]
            self.eval_baseline = np.einsum(
                "tk,tkg->tg", weights, neighbor_expression, optimize=True
            ).astype(np.float32).T

    def set_epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self):
        return self.masks_per_epoch if self.split == "train" else 1

    def _training_targets(self, index: int) -> np.ndarray:
        count = max(
            1,
            round(
                len(self.data.observed_spots) * self.train_target_fraction
            ),
        )
        # SeedSequence avoids collisions between epochs and sample indices and
        # remains deterministic regardless of DataLoader iteration order.
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, index])
        )
        targets = rng.choice(
            self.data.observed_spots, size=count, replace=False
        )
        return np.sort(targets.astype(np.int64, copy=False))

    def __getitem__(self, index: int):
        if self.split == "train":
            target_spots = self._training_targets(index)
            visible_lookup = np.ones(
                len(self.data.observed_spots), dtype=bool
            )
            hidden = np.isin(
                self.data.observed_spots, target_spots, assume_unique=True
            )
            visible_lookup[hidden] = False
            visible_spots = self.data.observed_spots[visible_lookup]
            baseline = np.zeros(
                (self.n_genes, len(target_spots)), dtype=np.float32
            )
        else:
            target_spots = self.eval_targets
            visible_spots = self.data.observed_spots
            baseline = self.eval_baseline

        input_mask = np.zeros(
            (1, self.height, self.width), dtype=np.float32
        )
        input_mask[
            0,
            self.data.spot_rows[visible_spots],
            self.data.spot_cols[visible_spots],
        ] = 1.0
        # Multiplication hides every gene at a target spot simultaneously.
        inp = self.expression_grid * input_mask
        target_values = self.data.expression[target_spots].T.astype(
            np.float32, copy=False
        )
        target_indices = (
            self.data.spot_rows[target_spots] * self.width
            + self.data.spot_cols[target_spots]
        ).astype(np.int64, copy=False)

        return {
            "inp": torch.from_numpy(inp),
            "input_mask": torch.from_numpy(input_mask),
            "target_values": torch.from_numpy(target_values),
            "target_indices": torch.from_numpy(target_indices),
            "baseline": torch.from_numpy(baseline),
        }


class PointJointMaskedVisiumDataset(Dataset):
    """Grid-free joint samples made only of visible and query points.

    Every item is one spatial domain under one random visibility mask. No
    raster tensor, flattened grid index, or padding spot is constructed.
    """

    def __init__(
        self,
        data: MaskedVisiumData,
        split: str,
        masks_per_epoch: int = 64,
        train_target_fraction: float = 0.25,
        idw_neighbors: int = 6,
        seed: int = 2026,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if masks_per_epoch < 1:
            raise ValueError("masks_per_epoch must be positive")
        if not 0.0 < train_target_fraction < 1.0:
            raise ValueError("train_target_fraction must be between 0 and 1")
        if idw_neighbors < 1:
            raise ValueError("idw_neighbors must be positive")
        if not np.isfinite(data.expression).all():
            raise ValueError("expression contains non-finite values")
        if not np.isfinite(data.physical_xy).all():
            raise ValueError("physical coordinates contain non-finite values")

        self.data = data
        self.split = split
        self.masks_per_epoch = masks_per_epoch
        self.train_target_fraction = train_target_fraction
        self.idw_neighbors = idw_neighbors
        self.seed = seed
        self.epoch = 0

        self.eval_targets = None
        self.eval_baseline = None
        if split != "train":
            self.eval_targets = (
                data.validation_spots if split == "val" else data.test_spots
            )
            self.eval_baseline = self._idw_baseline(
                data.observed_spots, self.eval_targets
            )

    def set_epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self):
        return self.masks_per_epoch if self.split == "train" else 1

    def _training_targets(self, index: int) -> np.ndarray:
        count = max(
            1,
            round(
                len(self.data.observed_spots) * self.train_target_fraction
            ),
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, index])
        )
        targets = rng.choice(
            self.data.observed_spots, size=count, replace=False
        )
        return np.sort(targets.astype(np.int64, copy=False))

    def _idw_baseline(
        self,
        visible_spots: np.ndarray,
        target_spots: np.ndarray,
    ) -> np.ndarray:
        neighbors = min(self.idw_neighbors, len(visible_spots))
        tree = cKDTree(self.data.physical_xy[visible_spots])
        distances, indices = tree.query(
            self.data.physical_xy[target_spots], k=neighbors
        )
        if neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, 1e-6) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        neighbor_expression = self.data.expression[visible_spots[indices]]
        return np.einsum(
            "qk,qkg->qg", weights, neighbor_expression, optimize=True
        ).astype(np.float32)

    def __getitem__(self, index: int):
        if self.split == "train":
            target_spots = self._training_targets(index)
            hidden = np.isin(
                self.data.observed_spots, target_spots, assume_unique=True
            )
            visible_spots = self.data.observed_spots[~hidden]
            # Training does not report the external baseline. Keeping its
            # correct point shape avoids an unnecessary CPU KNN per mask.
            baseline = np.zeros(
                (len(target_spots), len(self.data.genes)), dtype=np.float32
            )
        else:
            target_spots = self.eval_targets
            visible_spots = self.data.observed_spots
            baseline = self.eval_baseline

        return {
            "visible_expression": torch.from_numpy(
                self.data.expression[visible_spots].astype(
                    np.float32, copy=False
                )
            ),
            "visible_coord": torch.from_numpy(
                self.data.physical_xy[visible_spots].astype(
                    np.float32, copy=False
                )
            ),
            "query_coord": torch.from_numpy(
                self.data.physical_xy[target_spots].astype(
                    np.float32, copy=False
                )
            ),
            "target_values": torch.from_numpy(
                self.data.expression[target_spots].astype(
                    np.float32, copy=False
                )
            ),
            "target_spots": torch.from_numpy(
                target_spots.astype(np.int64, copy=False)
            ),
            "visible_spots": torch.from_numpy(
                visible_spots.astype(np.int64, copy=False)
            ),
            "baseline": torch.from_numpy(baseline),
        }
