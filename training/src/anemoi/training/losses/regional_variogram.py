# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from __future__ import annotations

import logging
from collections.abc import Sequence

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class RegionalVariogramScore(BaseLoss):
    """
    Regional variogram score for ensemble forecasts.

    The loss compares observed spatial increments against the
    ensemble-mean predicted spatial increments for selected grid offsets.

    Input shapes
    ------------
    y_pred:
        (batch, ensemble, latlon, variable)

    y_target:
        (batch, latlon, variable)
    """

    def __init__(
        self,
        field_shape: tuple[int, int] | list[int],
        offsets: Sequence[Sequence[int]] | None = None,
        offset_weights: Sequence[float] | None = None,
        variogram_power: float = 0.5,
        distance_weight_power: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        field_shape:
            Regional grid shape as ``(ydim, xdim)``.

        offsets:
            Spatial offsets ``(dy, dx)`` used to form grid-point pairs.

            For example:

            ``[(0, 1), (1, 0), (1, 1), (1, -1)]``

            compares nearest horizontal, vertical, and diagonal neighbors.

        offset_weights:
            Optional explicit weight for each offset. If omitted, weights
            are derived from spatial distance.

        variogram_power:
            Exponent ``p`` applied to absolute spatial differences.
            Common choices are 0.5 or 1.0.

        distance_weight_power:
            If explicit offset weights are not provided, the weight is

                1 / distance**distance_weight_power

            Set to 0 for equal offset weights.

        ignore_nans:
            Ignore pairs containing NaNs or infinities.
        """
        super().__init__(
            ignore_nans=ignore_nans,
            **kwargs,
        )

        # Supports tuples, lists, and Hydra/OmegaConf ListConfig.
        try:
            field_shape = tuple(field_shape)
        except TypeError as exc:
            raise ValueError(
                "field_shape must contain (ydim, xdim)."
            ) from exc

        if len(field_shape) != 2:
            raise ValueError(
                "field_shape must contain exactly two values: "
                "(ydim, xdim)."
            )

        ydim, xdim = field_shape

        if (
            not isinstance(ydim, int)
            or isinstance(ydim, bool)
            or ydim <= 0
        ):
            raise ValueError(
                f"ydim must be a positive integer, got {ydim!r}."
            )

        if (
            not isinstance(xdim, int)
            or isinstance(xdim, bool)
            or xdim <= 0
        ):
            raise ValueError(
                f"xdim must be a positive integer, got {xdim!r}."
            )

        if not 0.0 < variogram_power <= 2.0:
            raise ValueError(
                "variogram_power must be in (0, 2], "
                f"got {variogram_power}."
            )

        if distance_weight_power < 0.0:
            raise ValueError(
                "distance_weight_power must be non-negative, "
                f"got {distance_weight_power}."
            )

        self.ydim = ydim
        self.xdim = xdim
        self.len_reg = ydim * xdim

        self.variogram_power = float(variogram_power)
        self.distance_weight_power = float(distance_weight_power)

        if offsets is None:
            # A modest multiscale default.
            offsets = (
                (0, 1),
                (1, 0),
                (1, 1),
                (1, -1),
                (0, 2),
                (2, 0),
                (2, 2),
                (2, -2),
                (0, 4),
                (4, 0),
                (4, 4),
                (4, -4),
                (0, 8),
                (8, 0),
            )

        parsed_offsets: list[tuple[int, int]] = []

        for offset in offsets:
            offset = tuple(offset)

            if len(offset) != 2:
                raise ValueError(
                    f"Each offset must contain (dy, dx), got {offset!r}."
                )

            dy, dx = offset

            if (
                not isinstance(dy, int)
                or isinstance(dy, bool)
                or not isinstance(dx, int)
                or isinstance(dx, bool)
            ):
                raise ValueError(
                    f"Offsets must contain integers, got {offset!r}."
                )

            if dy == 0 and dx == 0:
                raise ValueError(
                    "Offset (0, 0) is not a valid variogram pair."
                )

            if abs(dy) >= self.ydim or abs(dx) >= self.xdim:
                raise ValueError(
                    f"Offset {offset!r} does not fit inside "
                    f"field_shape={field_shape}."
                )

            parsed_offsets.append((dy, dx))

        if not parsed_offsets:
            raise ValueError(
                "At least one spatial offset must be provided."
            )

        self.offsets = tuple(parsed_offsets)

        if offset_weights is not None:
            offset_weights = tuple(float(x) for x in offset_weights)

            if len(offset_weights) != len(self.offsets):
                raise ValueError(
                    "offset_weights must have the same length as offsets."
                )

            if any(weight < 0.0 for weight in offset_weights):
                raise ValueError(
                    "offset_weights must be non-negative."
                )

            weights = torch.tensor(
                offset_weights,
                dtype=torch.float32,
            )
        else:
            distances = torch.tensor(
                [
                    (dy * dy + dx * dx) ** 0.5
                    for dy, dx in self.offsets
                ],
                dtype=torch.float32,
            )

            if self.distance_weight_power == 0.0:
                weights = torch.ones_like(distances)
            else:
                weights = distances.pow(
                    -self.distance_weight_power
                )

        if not bool((weights > 0).any()):
            raise ValueError(
                "At least one variogram offset weight must be positive."
            )

        # Normalize so changing the number of offsets does not automatically
        # change the overall loss scale.
        weights = weights / weights.sum().clamp_min(1.0e-12)

        self.register_buffer(
            "offset_weights",
            weights,
            persistent=True,
        )

    @staticmethod
    def _paired_slices(
        dy: int,
        dx: int,
    ) -> tuple[
        tuple[slice, slice],
        tuple[slice, slice],
    ]:
        """
        Construct two spatial slices separated by ``(dy, dx)``.

        No padding or periodic wrapping is used.
        """
        if dy >= 0:
            y_slice_a = slice(0, -dy if dy > 0 else None)
            y_slice_b = slice(dy, None)
        else:
            y_slice_a = slice(-dy, None)
            y_slice_b = slice(0, dy)

        if dx >= 0:
            x_slice_a = slice(0, -dx if dx > 0 else None)
            x_slice_b = slice(dx, None)
        else:
            x_slice_a = slice(-dx, None)
            x_slice_b = slice(0, dx)

        return (
            (y_slice_a, x_slice_a),
            (y_slice_b, x_slice_b),
        )

    def _variogram_score(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the sampled regional variogram score.

        Parameters
        ----------
        preds:
            Shape ``(batch, variable, ensemble, y, x)``.

        targets:
            Shape ``(batch, variable, y, x)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, variable)``.
        """
        if preds.shape[-2:] != (self.ydim, self.xdim):
            raise ValueError(
                "Prediction spatial shape does not match field_shape: "
                f"got {tuple(preds.shape[-2:])}, expected "
                f"{(self.ydim, self.xdim)}."
            )

        if targets.shape[-2:] != (self.ydim, self.xdim):
            raise ValueError(
                "Target spatial shape does not match field_shape: "
                f"got {tuple(targets.shape[-2:])}, expected "
                f"{(self.ydim, self.xdim)}."
            )

        preds = preds.float()
        targets = targets.float()

        batch_size = preds.shape[0]
        n_variables = preds.shape[1]

        total_score = torch.zeros(
            (batch_size, n_variables),
            device=preds.device,
            dtype=preds.dtype,
        )

        total_weight = torch.zeros(
            (batch_size, n_variables),
            device=preds.device,
            dtype=preds.dtype,
        )

        offset_weights = self.offset_weights.to(
            device=preds.device,
            dtype=preds.dtype,
        )

        for offset_index, (dy, dx) in enumerate(self.offsets):
            slices_a, slices_b = self._paired_slices(
                dy=dy,
                dx=dx,
            )

            ya, xa = slices_a
            yb, xb = slices_b

            # Prediction pair differences:
            # (batch, variable, ensemble, pair_y, pair_x)
            pred_a = preds[..., ya, xa]
            pred_b = preds[..., yb, xb]

            # Target pair differences:
            # (batch, variable, pair_y, pair_x)
            target_a = targets[..., ya, xa]
            target_b = targets[..., yb, xb]

            pred_valid = (
                torch.isfinite(pred_a)
                & torch.isfinite(pred_b)
            ).all(dim=2)

            target_valid = (
                torch.isfinite(target_a)
                & torch.isfinite(target_b)
            )

            pair_valid = pred_valid & target_valid

            if not self.ignore_nans and not bool(pair_valid.all()):
                raise ValueError(
                    "Prediction or target contains NaN/Inf in "
                    f"variogram pairs for offset {(dy, dx)}."
                )

            pred_a = torch.nan_to_num(
                pred_a,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            pred_b = torch.nan_to_num(
                pred_b,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            target_a = torch.nan_to_num(
                target_a,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            target_b = torch.nan_to_num(
                target_b,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Ensemble-mean predicted variogram increment.
            pred_increment = (
                pred_a - pred_b
            ).abs().pow(
                self.variogram_power
            ).mean(
                dim=2
            )

            observed_increment = (
                target_a - target_b
            ).abs().pow(
                self.variogram_power
            )

            pair_error = (
                observed_increment - pred_increment
            ).square()

            pair_valid_float = pair_valid.to(
                dtype=pair_error.dtype
            )

            valid_count = pair_valid_float.sum(
                dim=(-2, -1)
            )

            offset_score = (
                pair_error * pair_valid_float
            ).sum(
                dim=(-2, -1)
            ) / valid_count.clamp_min(1.0)

            sample_has_valid_pairs = valid_count > 0

            offset_weight = offset_weights[offset_index]

            total_score = (
                total_score
                + offset_weight
                * offset_score
                * sample_has_valid_pairs
            )

            total_weight = (
                total_weight
                + offset_weight
                * sample_has_valid_pairs
            )

        if bool((total_weight <= 0).any()):
            raise ValueError(
                "At least one batch-variable sample has no valid "
                "variogram pairs."
            )

        return total_score / total_weight.clamp_min(1.0e-12)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        del squash, group

        if grid_shard_slice is not None:
            raise AssertionError(
                "Set 'keep_batch_sharded=False' in the model "
                "configuration to compute the regional variogram score."
            )

        if y_pred.ndim != 4:
            raise ValueError(
                "y_pred must have shape "
                "(batch, ensemble, latlon, variable)."
            )

        if y_target.ndim != 3:
            raise ValueError(
                "y_target must have shape "
                "(batch, latlon, variable)."
            )

        if y_pred.shape[2] < self.len_reg:
            raise ValueError(
                f"y_pred has {y_pred.shape[2]} nodes, but "
                f"{self.len_reg} regional nodes are required."
            )

        if y_target.shape[1] < self.len_reg:
            raise ValueError(
                f"y_target has {y_target.shape[1]} nodes, but "
                f"{self.len_reg} regional nodes are required."
            )

        y_pred_regional = y_pred[
            :,
            :,
            : self.len_reg,
        ]

        y_target_regional = y_target[
            :,
            : self.len_reg,
        ]

        # Prediction:
        # (batch, ensemble, latlon, variable)
        # -> (batch, variable, ensemble, y, x)
        y_pred_regional = einops.rearrange(
            y_pred_regional,
            "bs e (y x) v -> bs v e y x",
            y=self.ydim,
            x=self.xdim,
        )

        # Target:
        # (batch, latlon, variable)
        # -> (batch, variable, y, x)
        y_target_regional = einops.rearrange(
            y_target_regional,
            "bs (y x) v -> bs v y x",
            y=self.ydim,
            x=self.xdim,
        )

        score = self._variogram_score(
            y_pred_regional,
            y_target_regional,
        )

        # Match Anemoi's common loss/scaler layout:
        # (batch, variable) -> (batch, 1, 1, variable)
        score = einops.rearrange(
            score,
            "bs v -> bs 1 1 v",
        )

        scaled = self.scale(
            score,
            scaler_indices,
            without_scalers=without_scalers,
        )

        return scaled.mean()

    @property
    def name(self) -> str:
        return (
            f"regional-variogram-p"
            f"{self.variogram_power:.2f}"
        )
