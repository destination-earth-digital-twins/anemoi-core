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

# from torch_dct import dct_2d
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.kcrps import AlmostFairKernelCRPS

LOGGER = logging.getLogger(__name__)


class AFCRPSFFTLossNew(AlmostFairKernelCRPS):
    """Almost-fair kernel CRPS computed on local Fourier coefficients."""

    def __init__(
        self,
        field_shape: tuple[int, int],
        cutoff_ratio: float = 1.0,
        alpha: float = 1.0,
        local: int = 1,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        frequency_beta: float = 2.0,
        frequency_power: float = 1.5,
        apply_window: bool = True,
        apply_frequency_weight: bool = True,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        field_shape:
            Spatial field shape ``(ydim, xdim)``.
        cutoff_ratio:
            Fraction of the maximum radial frequency to retain. Must be in
            ``(0, 1]``.
        alpha:
            Almost-fair CRPS parameter.
        local:
            Number of non-overlapping FFT patches along each spatial axis.
            The total number of patches is ``local**2``.
        no_autocast:
            Disable autocast during FFT and CRPS computation.
        ignore_nans:
            Use NaN-aware loss reductions where supported by the parent class.
        frequency_beta:
            Strength of radial frequency weighting. A value of zero gives
            uniform weighting.
        frequency_power:
            Exponent used in the radial frequency weight.
        apply_window:
            Apply an energy-normalized Hann window to each patch.
        apply_frequency_weight:
            Apply radial weighting to the spectral CRPS.
        """
        super().__init__(
            alpha=alpha,
            ignore_nans=ignore_nans,
            **kwargs,
        )
        field_shape = tuple(field_shape)

        if len(field_shape) != 2:
            raise ValueError(
                "field_shape must contain exactly two integers: (ydim, xdim)."
            )

        ydim, xdim = field_shape

        if not isinstance(ydim, int) or ydim <= 0:
            raise ValueError(f"ydim must be a positive integer, got {ydim!r}.")

        if not isinstance(xdim, int) or xdim <= 0:
            raise ValueError(f"xdim must be a positive integer, got {xdim!r}.")

        if not isinstance(local, int) or local <= 0:
            raise ValueError(f"local must be a positive integer, got {local!r}.")

        if ydim % local != 0 or xdim % local != 0:
            raise ValueError(
                "The field dimensions must be divisible by local for "
                "non-overlapping patches. "
                f"Got field_shape={field_shape} and local={local}."
            )

        if not 0.0 < cutoff_ratio <= 1.0:
            raise ValueError(
                "cutoff_ratio must be in the interval (0, 1], " f"got {cutoff_ratio}."
            )

        if frequency_beta < 0:
            raise ValueError(
                f"frequency_beta must be non-negative, got {frequency_beta}."
            )

        if frequency_power <= 0:
            raise ValueError(
                f"frequency_power must be positive, got {frequency_power}."
            )

        self.ydim = ydim
        self.xdim = xdim
        self.len_reg = ydim * xdim

        self.local = local
        self.ydim_local = ydim // local
        self.xdim_local = xdim // local

        self.cutoff_ratio = cutoff_ratio
        self.no_autocast = no_autocast
        self.apply_window = apply_window
        self.apply_frequency_weight = apply_frequency_weight
        self.transform = torch.fft.rfft2

        self.register_buffer(
            "frequency_weight",
            self.frequency_weight_2d(
                nx=self.xdim_local,
                ny=self.ydim_local,
                beta=frequency_beta,
                power=frequency_power,
                cutoff_ratio=cutoff_ratio,
            ),
        )

        if apply_window:
            wy = torch.hann_window(
                self.ydim_local,
                periodic=False,
                dtype=torch.float32,
            )
            wx = torch.hann_window(
                self.xdim_local,
                periodic=False,
                dtype=torch.float32,
            )

            window = torch.outer(wy, wx)

            # Keep the mean squared window amplitude approximately equal to 1.
            rms = window.square().mean().sqrt().clamp_min(1e-8)
            window = window / rms
        else:
            window = torch.ones(
                self.ydim_local,
                self.xdim_local,
                dtype=torch.float32,
            )

        self.register_buffer("window", window)

    @staticmethod
    def frequency_weight_2d(
        nx: int,
        ny: int,
        beta: float = 1.0,
        power: float = 2.0,
        cutoff_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Construct radial weights matching an ``rfft2`` output.

        An ``rfft2`` of an ``(ny, nx)`` field has shape

        ``(ny, nx // 2 + 1)``.

        Only the final dimension uses ``rfftfreq``.
        """
        fy = torch.fft.fftfreq(ny)
        fx = torch.fft.rfftfreq(nx)

        ky, kx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(kx.square() + ky.square())

        max_radius = radius.max().clamp_min(1e-8)
        normalized_radius = radius / max_radius

        weight = 1.0 + beta * normalized_radius.pow(power)

        if cutoff_ratio < 1.0:
            mask = normalized_radius <= cutoff_ratio
            weight = weight * mask

        return weight[None, None, ...]

    def _discrete_transform(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Transform local patches and evaluate coefficient-wise AFCRPS.

        Parameters
        ----------
        preds:
            Prediction tensor with shape ``(batch*variables, ensemble, y, x)``.
        targets:
            Target tensor with shape ``(batch*variables, y, x)``.
        batch_size:
            Original batch size.

        Returns
        -------
        torch.Tensor
            Loss tensor with shape ``(batch, variables, 1)``.
        """
        expected_pred_shape = (
            self.ydim,
            self.xdim,
        )

        if preds.shape[-2:] != expected_pred_shape:
            raise ValueError(
                f"Expected prediction spatial shape {expected_pred_shape}, "
                f"got {tuple(preds.shape[-2:])}."
            )

        if targets.shape[-2:] != expected_pred_shape:
            raise ValueError(
                f"Expected target spatial shape {expected_pred_shape}, "
                f"got {tuple(targets.shape[-2:])}."
            )

        # FFT support is more reliable in float32 than in float16/bfloat16,
        # particularly for arbitrary spatial dimensions.
        preds = preds.float()
        targets = targets.float()

        # Non-overlapping local patches:
        #
        # preds:
        #   (batch*variables, ensemble, y, x)
        #   -> (batch*variables, ensemble, local**2, patch_y, patch_x)
        #
        # targets:
        #   (batch*variables, y, x)
        #   -> (batch*variables, local**2, patch_y, patch_x)
        preds_local = einops.rearrange(
            preds,
            "bv e (ly py) (lx px) -> bv e (ly lx) py px",
            ly=self.local,
            lx=self.local,
        )

        targets_local = einops.rearrange(
            targets,
            "bv (ly py) (lx px) -> bv (ly lx) py px",
            ly=self.local,
            lx=self.local,
        )

        window = self.window.to(
            device=preds_local.device,
            dtype=preds_local.dtype,
        )

        preds_local = preds_local * window
        targets_local = targets_local * window

        # rfft2 keeps all y frequencies but only the nonredundant x frequencies:
        #
        # (..., patch_y, patch_x)
        # -> (..., patch_y, patch_x // 2 + 1)
        preds_spectral = self.transform(
            preds_local,
            dim=(-2, -1),
            norm="ortho",
        )

        targets_spectral = self.transform(
            targets_local,
            dim=(-2, -1),
            norm="ortho",
        )

        preds_spectral = einops.rearrange(
            preds_spectral,
            "(bs v) e l ky kx -> bs v (l ky kx) e",
            bs=batch_size,
        )

        targets_spectral = einops.rearrange(
            targets_spectral,
            "(bs v) l ky kx -> bs v (l ky kx)",
            bs=batch_size,
        )

        # Expected output: (batch, variables, spectral_coefficients)
        kcrps = self._kernel_crps(
            preds_spectral,
            targets_spectral,
            self.alpha,
        )

        if not self.apply_frequency_weight:
            return kcrps.mean(dim=-1, keepdim=True)

        # The same frequency weights apply independently to every local patch.
        frequency_weight = einops.repeat(
            self.frequency_weight,
            "1 1 ky kx -> 1 1 (l ky kx)",
            l=self.local**2,
        ).to(
            device=kcrps.device,
            dtype=kcrps.real.dtype,
        )

        denominator = frequency_weight.sum(dim=-1, keepdim=True)

        if torch.any(denominator <= 0):
            raise RuntimeError(
                "The spectral mask removed all frequencies. Increase cutoff_ratio."
            )

        return (kcrps * frequency_weight).sum(dim=-1, keepdim=True) / denominator

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
            raise ValueError(
                "Set 'keep_batch_sharded=False' in the model configuration "
                "to compute the spectral loss."
            )

        batch_size = y_pred.shape[0]

        if y_pred.shape[2] < self.len_reg:
            raise ValueError(
                f"y_pred contains {y_pred.shape[2]} grid points, but "
                f"{self.len_reg} are required by field_shape."
            )

        if y_target.shape[1] < self.len_reg:
            raise ValueError(
                f"y_target contains {y_target.shape[1]} grid points, but "
                f"{self.len_reg} are required by field_shape."
            )

        y_pred_regional = y_pred[:, :, : self.len_reg]
        y_target_regional = y_target[:, : self.len_reg]

        y_pred_regional = einops.rearrange(
            y_pred_regional,
            "bs e (y x) v -> (bs v) e y x",
            y=self.ydim,
            x=self.xdim,
        )

        y_target_regional = einops.rearrange(
            y_target_regional,
            "bs (y x) v -> (bs v) y x",
            y=self.ydim,
            x=self.xdim,
        )

        if self.no_autocast:
            # Use the actual device type instead of hard-coding CUDA.
            device_type = y_pred_regional.device.type

            with torch.autocast(
                device_type=device_type,
                enabled=False,
            ):
                kcrps = self._discrete_transform(
                    y_pred_regional,
                    y_target_regional,
                    batch_size,
                )
        else:
            kcrps = self._discrete_transform(
                y_pred_regional,
                y_target_regional,
                batch_size,
            )

        kcrps = einops.rearrange(
            kcrps,
            "bs v spectral -> bs 1 spectral v",
        )

        scaled = self.scale(
            kcrps,
            scaler_indices,
            without_scalers=without_scalers,
        )
        return scaled.mean()

    @property
    def name(self) -> str:
        return "CRPS-FFT"
