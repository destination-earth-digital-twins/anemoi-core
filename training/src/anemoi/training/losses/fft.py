# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property

import einops
import torch
#from torch_dct import dct_2d
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.kcrps import KernelCRPS
from anemoi.training.losses.kcrps import AlmostFairKernelCRPS

LOGGER = logging.getLogger(__name__)

class CRPSFFTLoss(KernelCRPS):
    """CRPS computed in spectral space, like in FourCastNet3
    """

    def __init__(
        self,
        field_shape: tuple[int, int],
        cutoff_ratio: float = 1.0,
        fft: bool = False,
        alpha: float = 1.0,  # remove this
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Parameters
        ----------
        xdim: int
            Shape of regional domain to apply transform on, x component
        ydim: int
            Shape of regional domain to apply transform on, y component
        fft: bool, optional
            Do the Fourier transform instead of discrete cosine, by default False
        eps: float, optional
            Normalizing factor for transformed field for numerical stability
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        assert field_shape is not None, f"field_shape is None. Can only process grids with field shape with the form XY shape" #len(field_shape) == 2 and isinstance(field_shape, (tuple,list)), "dim should be a tuple of two integers"
        ydim,xdim = field_shape
        
        assert xdim > 0 and isinstance(xdim, int), f"xdim must be positive integer. Got {xdim}"
        assert ydim > 0 and isinstance(ydim, int), f"ydim must be positive integer. Got {ydim}"

        self.xdim = xdim
        self.ydim = ydim
        self.len_reg = xdim * ydim
        #self.transform = torch.fft.fft2 if fft else dct_2d
        self.mask = self.lowpass_mask_2d(xdim, ydim, cutoff_ratio)
        self.no_autocast = no_autocast

    @staticmethod
    def lowpass_mask_2d(nx, ny, cutoff_ratio=1.0, *,
                        dx=1.0, dy=1.0,
                        shifted=False):
        """
        Create a circular low-pass mask for a 2D FFT grid.

        Args:
            nx, ny: spatial sizes.
            cutoff_ratio: 0..1, relative to the radial Nyquist (1 is as large as allowed).
            dx, dy: sample spacing in x and y (units per pixel). Default 1.
            shifted: if True, mask is centered (for use with fftshifted spectra).
                     if False, mask matches unshifted torch.fft.fft2 output.

        Returns:
            mask of shape (nx, ny), True inside the passband.
        """
        # Frequency grids in cycles per unit
        fx = torch.fft.fftfreq(nx, d=dx)
        fy = torch.fft.fftfreq(ny, d=dy)
        KX, KY = torch.meshgrid(fx, fy, indexing='ij')

        # Radial spatial frequency
        k = torch.sqrt(KX*KX + KY*KY)

        # Convert ratio -> absolute cutoff in cycles per unit
        fx_nyq = 1.0 / (2.0 * dx)
        fy_nyq = 1.0 / (2.0 * dy)
        k_nyq_radial = min(fx_nyq, fy_nyq)  # circular limit
        k_cut = float(cutoff_ratio) * k_nyq_radial

        mask = torch.where(k < k_cut, 1.0 - 2.0 * k, 0.0)

        # Align to shifted spectrum if requested
        if shifted:
            mask = torch.fft.fftshift(mask, dim=(-2, -1))

        mask = einops.rearrange(mask, "x y -> 1 1 (y x)")
        return mask

    def _discrete_transform(self, preds: torch.Tensor, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Perform the discrete Fourier/cosine transform of preds and targets and return log-diff.

        Args:
            preds: torch.Tensor
                Predictions, (bs*var, ens, y, x)
            targets: torch.Tensor
                Targets, (bs*var, y, x)
            batch_size: int
                Self-explanatory
        """

        preds_spectral = self.transform(preds)
        targets_spectral = self.transform(targets)

        preds_spectral = einops.rearrange(
                preds_spectral,
                "(bs v) e y x -> bs v (y x) e",
                bs=batch_size,
        )
        targets_spectral = einops.rearrange(
                targets_spectral,
                "(bs v) y x -> bs v (y x)",
                bs=batch_size,
        )

        #mask = einops.rearrange(
        #        self.mask.to(preds.device),
        #        "y x -> 1 1 (y x)",
        #)

        kcrps_ = self._kernel_crps(preds_spectral, targets_spectral)
        return kcrps_ * self.mask.to(preds.device)


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
        assert not is_sharded, "Set 'keep_batch_sharded=False in the model config to compute spectral loss"

        bs_ = y_pred.shape[0]  # batch size

        y_pred_regional = y_pred[:, :, :self.len_reg]
        y_target_regional = y_target[:, :self.len_reg]
        
        y_pred_regional = einops.rearrange(
                y_pred_regional,
                "bs e (y x) v -> (bs v) e y x",
                x=self.xdim,
                y=self.ydim,
        )
        y_target_regional = einops.rearrange(
                y_target_regional, 
                "bs (y x) v -> (bs v) y x",
                x=self.xdim,
                y=self.ydim,
        )

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)
        else:
            kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)

        #torch.save(log_diff, "/leonardo/home/userexternal/enordhag/log_diff.pt")

        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        scaled = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers)
        print("FFT loss:", scaled.mean())
        return scaled.mean() #self.reduce(kcrps_, squash=squash, squash_mode="avg", group=None)

    @property
    def name(self) -> str:
        return "CRPS-FFT"

class AFCRPSFFTLoss(AlmostFairKernelCRPS):
    """CRPS computed in spectral space, like in FourCastNet3
    """

    def __init__(
        self,
        field_shape: tuple[int, int],
        cutoff_ratio: float = 1.0,
        alpha: float = 1.0,
        local: int = 1, #3,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Latitude- and (inverse-)variance-weighted kernel CRPS loss.

        Parameters
        ----------
        xdim: int
            Shape of regional domain to apply transform on, x component
        ydim: int
            Shape of regional domain to apply transform on, y component
        cutoff_ratio: float
            Cutoff ratio used in mask
        alpha: float
            Weight factor between fair and normal crps
        local: int
            Number of localized windows for FFT in each direction
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(alpha=alpha, ignore_nans=ignore_nans, **kwargs)
        
        #assert len(field_shape) == 2 and isinstance(field_shape, [tuple,list]), "dim should be a tuple of two integers"
        assert field_shape is not None, f"field_shape is None. Can only process grids with field shape with the form XY shape"
        ydim,xdim = field_shape

        assert xdim > 0 and isinstance(xdim, int), f"xdim must be positive integer. Got {xdim}"
        assert ydim > 0 and isinstance(ydim, int), f"ydim must be positive integer. Got {ydim}"

        self.xdim = xdim
        self.ydim = ydim
        self.len_reg = xdim * ydim
        self.local = local
        self.xdim_local = xdim // local
        self.ydim_local = ydim // local
        self.transform = torch.fft.fft2
        self.mask = self.lowpass_mask_2d(self.xdim_local, self.ydim_local, cutoff_ratio)
        self.no_autocast = no_autocast

    @staticmethod
    def lowpass_mask_2d(nx, ny, cutoff_ratio=1.0, *,
                        dx=1.0, dy=1.0,
                        shifted=False):
        """
        Create a circular low-pass mask for a 2D FFT grid.

        Args:
            nx, ny: spatial sizes.
            cutoff_ratio: 0..1, relative to the radial Nyquist (1 is as large as allowed).
            dx, dy: sample spacing in x and y (units per pixel). Default 1.
            shifted: if True, mask is centered (for use with fftshifted spectra).
                     if False, mask matches unshifted torch.fft.fft2 output.

        Returns:
            mask of shape (nx, ny), True inside the passband.
        """
        # Frequency grids in cycles per unit
        fx = torch.fft.fftfreq(nx, d=dx)
        fy = torch.fft.fftfreq(ny, d=dy)
        KX, KY = torch.meshgrid(fx, fy, indexing='ij')

        # Radial spatial frequency
        k = torch.sqrt(KX*KX + KY*KY)

        # Convert ratio -> absolute cutoff in cycles per unit
        fx_nyq = 1.0 / (2.0 * dx)
        fy_nyq = 1.0 / (2.0 * dy)
        k_nyq_radial = min(fx_nyq, fy_nyq)  # circular limit
        k_cut = float(cutoff_ratio) * k_nyq_radial

        mask = (k < k_cut) #torch.where(k < k_cut, 1.0 - 2.0 * k, 0.0)

        # Align to shifted spectrum if requested
        if shifted:
            mask = torch.fft.fftshift(mask, dim=(-2, -1))

        mask = einops.rearrange(mask, "x y -> 1 1 y x")
        return mask

    def _discrete_transform(self, preds: torch.Tensor, targets: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Perform the discrete Fourier/cosine transform of preds and targets and return log-diff.

        Args:
            preds: torch.Tensor
                Predictions, (bs*var, ens, y, x)
            targets: torch.Tensor
                Targets, (bs*var, y, x)
            batch_size: int
                Self-explanatory
        """
        bsvar = preds.shape[0]
        ens_size = preds.shape[1]
        nlocal = self.local * self.local

        # Reorganize predictions and targets from (y, x) to (local*local, y, x)
        preds_local = preds.reshape(bsvar, ens_size, self.local, self.ydim_local, self.local, self.xdim_local)
        preds_local = preds_local.swapaxes(3, 4)
        preds_local = preds_local.reshape(bsvar, ens_size, nlocal, self.ydim_local, self.xdim_local)
        targets_local = targets.reshape(bsvar, self.local, self.ydim_local, self.local, self.xdim_local)
        targets_local = targets_local.swapaxes(2, 3)
        targets_local = targets_local.reshape(bsvar, nlocal, self.ydim_local, self.xdim_local)

        # Perform transform on reorganized fields (localized FFT)
        preds_spectral = self.transform(preds_local)
        targets_spectral = self.transform(targets_local)

        # Normalize each patch individually (important, otherwise localized FFT has no effect on high frequencies)
        mean = targets_spectral.mean(axis=(0, 1))[None, None]
        #print("mean.shape:", mean.shape)
        #print("preds_spectral.shape:", preds_spectral.shape)
        #print("targets_spectral.shape:", targets_spectral.shape)
        preds_spectral /= mean[None]
        targets_spectral /= mean

        # Mask out frequencies higher than the Nyquist limit
        #print("self.mask.shape:", self.mask.shape)
        preds_spectral *= self.mask[None].to(preds.device)
        targets_spectral *= self.mask.to(preds.device)

        preds_spectral = einops.rearrange(
                preds_spectral,
                "(bs v) e l y x -> bs v (l y x) e",
                bs=batch_size,
        )
        targets_spectral = einops.rearrange(
                targets_spectral,
                "(bs v) l y x -> bs v (l y x)",
                bs=batch_size,
        )

        kcrps_ = self._kernel_crps(preds_spectral, targets_spectral, self.alpha)
        return kcrps_ #* self.mask.to(preds.device)


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
        assert not is_sharded, "Set 'keep_batch_sharded=False' in the model config to compute spectral loss"

        bs_ = y_pred.shape[0]  # batch size

        y_pred_regional = y_pred[:, :, :self.len_reg]
        y_target_regional = y_target[:, :self.len_reg]
        
        y_pred_regional = einops.rearrange(
                y_pred_regional,
                "bs e (y x) v -> (bs v) e y x",
                x=self.xdim,
                y=self.ydim,
        )
        y_target_regional = einops.rearrange(
                y_target_regional, 
                "bs (y x) v -> (bs v) y x",
                x=self.xdim,
                y=self.ydim,
        )

        if self.no_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)
        else:
            kcrps_ = self._discrete_transform(y_pred_regional, y_target_regional, bs_)

        kcrps_ = einops.rearrange(kcrps_, "bs v latlon -> bs 1 latlon v")
        scaled = self.scale(kcrps_, scaler_indices, without_scalers=without_scalers)
        return scaled.mean()

    @property
    def name(self) -> str:
        return "CRPS-FFT"