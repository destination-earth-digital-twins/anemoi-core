# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss

LOGGER = logging.getLogger(__name__)


class KernelCRPS(BaseLoss):
    """Kernel CRPS loss."""

    def __init__(
        self,
        fair: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Parameters
        ----------
        fair : bool
            Calculate a "fair" (unbiased) score - ensemble variance component weighted by (ens-size-1)^-1.
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        print("Initializing KernelCRPS loss")
        print(**kwargs)
        super().__init__(ignore_nans=ignore_nans, **kwargs)

        self.fair = fair

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, n_vars, latlon, ens_size)
        targets : torch.Tensor
            Ground truth, shape (batch_size, n_vars, latlon)

        Returns
        -------
        kCRPS : torch.Tensor
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """

        ens_size = preds.shape[-1]
        print("Preds shape: ", preds.shape)
        print(preds[0, 0, 0, :])
        mae = torch.mean(torch.abs(targets[..., None] - preds), dim=-1)

        assert ens_size > 1, "Ensemble size must be greater than 1."

        coef = -1.0 / (ens_size * (ens_size - 1)) if self.fair else -1.0 / (ens_size**2)

        ens_var = torch.zeros(size=preds.shape[:-1], device=preds.device)
        for i in range(ens_size):  # loop version to reduce memory usage
            ens_var += torch.sum(torch.abs(preds[..., i].unsqueeze(-1) - preds[..., i + 1 :]), dim=-1)
        ens_var = coef * ens_var
        print("MAE contribution ", mae)
        print("Ens variance contribution ", ens_var)
        return mae + ens_var

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
        is_sharded = grid_shard_slice is not None

        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs v latlon e")

        kcrps_ = self._kernel_crps(y_pred, y_target)

        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        kcrps_ = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(kcrps_, squash=squash, squash_mode="sum", group=group if is_sharded else None)

    @property
    def name(self) -> str:
        f_str = "f" if self.fair else ""
        return f"{f_str}kcrps"


class AlmostFairKernelCRPS(BaseLoss):
    """Almost fair kernel CRPS loss."""

    def __init__(
        self,
        alpha: float = 1.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        field_shape: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Parameters
        ----------
        alpha : float
            Factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)
        no_autocast : bool, optional
            Deactivate autocast for the kernel CRPS calculation
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)

        self.alpha = alpha
        self.no_autocast = no_autocast

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Kernel (ensemble) CRPS.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted ensemble, shape (batch_size, n_vars, latlon, ens_size)
        targets : torch.Tensor
            Ground truth, shape (batch_size, n_vars, latlon)
        alpha : float
            Factor for linear combination of fair (unbiased, ensemble variance component weighted by (ens-size-1)^-1)
            and standard CRPS (1.0 = fully fair, 0.0 = fully unfair)

        Returns
        -------
        kCRPS : torch.Tensor
            The point-wise kernel CRPS, shape (batch_size, 1, latlon).
        """
        ens_size = preds.shape[-1]
        print("Preds shape: ", preds.shape)
        print(preds[0, 0, 0, :])
        print("Targets shape: ", targets.shape)
        print(targets[0, 0, 0])

        epsilon = (1.0 - alpha) / ens_size

        var = torch.abs(preds.unsqueeze(dim=-1) - preds.unsqueeze(dim=-2))
        diag = torch.eye(ens_size, dtype=torch.bool, device=preds.device)
        err_r = einops.repeat(
            torch.abs(preds - targets.unsqueeze(dim=-1)),
            "batch var latlon ens -> batch var latlon n ens",
            n=ens_size,
        )
        print("STRD AVG PREDICTION member 0: ", torch.mean(preds[:,21,:,0]))
        print("STRD AVG PREDICTION member 1: ", torch.mean(preds[:,21,:,1]))
        print("STRD AVG TARGET: ", torch.mean(targets[:,21,:]))
        print("STRD MAE ERROR member 0: ", torch.abs(preds[:,21,:,0] - targets[:,21,:]).mean())
        print("STRD MAE ERROR member 1: ", torch.abs(preds[:,21,:,1] - targets[:,21,:]).mean())
        print('SSRD AVG PREDICTION member 0: ', torch.mean(preds[:,22,:,0]))
        print('SSRD AVG PREDICTION member 1: ', torch.mean(preds[:,22,:,1]))
        print('SSRD AVG TARGET: ', torch.mean(targets[:,22,:]))
        print('SSRD MAE ERROR member 0: ', torch.abs(preds[:,22,:,0] - targets[:,22,:]).mean())
        print('SSRD MAE ERROR member 1: ', torch.abs(preds[:,22,:,1] - targets[:,22,:]).mean())
        print('SURFACE PRESSURE member 0: ', torch.mean(preds[:,20,:,0]))
        print('SURFACE PRESSURE member 1: ', torch.mean(preds[:,20,:,1]))
        print('SURFACE PRESSURE TARGET: ', torch.mean(targets[:,20,:]))
        print('SURFACE PRESSURE MAE ERROR member 0: ', torch.abs(preds[:,20,:,0] - targets[:,20,:]).mean())
        print('SURFACE PRESSURE MAE ERROR member 1: ', torch.abs(preds[:,20,:,1] - targets[:,20,:]).mean())
        mem_err = err_r * ~diag
        mem_err_transpose = mem_err.transpose(-1, -2)
        print("err_r size: ", err_r.shape)
        print("err variable mean: ", torch.mean(err_r, dim=(0, 2, 3, 4)))
        print("mem_err size: ", mem_err.shape)
        print("mem_err mean: ", torch.mean(mem_err))
        print("mem_err variable mean: ", torch.mean(mem_err, dim=(0, 2, 3, 4)))
        print("var mean: ", torch.mean(var))
        assert ens_size > 1, "Ensemble size must be greater than 1."

        coef = 1.0 / (2.0 * ens_size * (ens_size - 1))
        print("coef: ", coef)
        print("epsilon: ", epsilon)
        return coef * torch.sum(mem_err + mem_err_transpose - (1 - epsilon) * var, dim=(-1, -2))

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
        is_sharded = grid_shard_slice is not None

        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs v latlon e")

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)
        else:
            kcrps_ = self._kernel_crps(y_pred, y_target, alpha=self.alpha)
        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        kcrps_ = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        print("Loss contribution from kernel CRPS: ", self.reduce(kcrps_, squash=squash, squash_mode="sum", group=group if is_sharded else None))
        return self.reduce(kcrps_, squash=squash, squash_mode="sum", group=group if is_sharded else None)

    @property
    def name(self) -> str:
        return f"afkcrps{self.alpha:.2f}"
       