import torch
import torch.nn.functional as F
from compressai.zoo import bmshj2018_factorized, mbt2018_mean
from compressai.models import CompressionModel
from hexplane.model.HexPlane_Base import HexPlane_Base
from typing import List, Tuple
import math


class HexPlane(HexPlane_Base):
    """
    A general version of HexPlane, which supports different fusion methods and feature regressor methods.
    """
    def __init__(self, aabb, gridSize, device, time_grid, near_far, **kargs):
        super().__init__(aabb, gridSize, device, time_grid, near_far, **kargs)
        self.use_codec = kargs.get('compression_enable', False)
        self.device = device

        if self.use_codec:
            self.init_codec()

    def init_codec(self):
        self.net_name = "mbt2018"
        self.app_feat = sum(self.app_n_comp)
        self.num_planes = len(self.app_plane)
        self.codec_block_size = 16

        # Initialize MBT2018 codec
        self.app_nets = torch.nn.ModuleList()
        for _ in range(self.num_planes):
            codec_net = mbt2018_mean(quality=2, pretrained=True)
            # Modify input channels if necessary
            # Assuming `update_input_channel` is a method you've defined
            # codec_net.update_input_channel(self.app_feat)
            self.app_nets.append(codec_net.to(self.device))

    def _pad(self, tensor):
        _, _, h, w = tensor.size()
        block_size = self.codec_block_size

        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size

        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        padded_tensor = F.pad(tensor, padding, mode='constant', value=0)
        return padded_tensor, padding

    def _unpad(self, tensor, padding):
        _, _, h, w = tensor.size()
        pad_top, pad_bottom, pad_left, pad_right = padding[2], padding[3], padding[0], padding[1]
        unpadded_tensor = tensor[:, :, pad_top:h - pad_bottom, pad_left:w - pad_right]
        return unpadded_tensor

    def encode_decode_app(self):
        out_planes = []
        out_rate = 0

        for idx in range(self.num_planes):
            # Implement compression logic
            compressed_plane = self.app_nets[idx](self.app_plane[idx])
            out_planes.append(compressed_plane)
            out_rate += self._estimate_rate(compressed_plane)

        self.app_plane = torch.nn.ParameterList(out_planes)
        self.app_rate = out_rate

        return {
            "app_plane": self.app_plane.state_dict(),
            "app_rate": self.app_rate.item(),
        }

    def _estimate_rate(self, output):
        num_pixels = output["x_hat"].numel() / output["x_hat"].shape[1]  # Assuming batch size is 1
        rate = torch.zeros(1, device=self.device)
        for likelihoods in output["likelihoods"].values():
            rate += torch.sum(-torch.log2(likelihoods)) / num_pixels
        return rate

    def _measure_size(self, compressed):
        size = 0
        for s in compressed["strings"]:
            for string in s:
                size += len(string)
        return size

    def init_planes(self, res, device):
        """
        Initialize the planes. density_plane is the spatial plane while density_line_time is the spatial-temporal plane.
        """
        self.density_plane, self.density_line_time = self.init_one_hexplane(
            self.density_n_comp, self.gridSize, device
        )
        self.app_plane, self.app_line_time = self.init_one_hexplane(
            self.app_n_comp, self.gridSize, device
        )

        if (
            self.fusion_two != "concat"
        ):  # if fusion_two is not concat, then we need dimensions from each paired planes are the same.
            assert self.app_n_comp[0] == self.app_n_comp[1]
            assert self.app_n_comp[0] == self.app_n_comp[2]

        # We use density_basis_mat and app_basis_mat to project extracted features from HexPlane to density_dim/app_dim.
        # density_basis_mat and app_basis_mat are linear layers, whose input dims are calculated based on the fusion methods.
        if self.fusion_two == "concat":
            if self.fusion_one == "concat":
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp) * 2, self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp) * 2, self.app_dim, bias=False
                ).to(device)
            else:
                self.density_basis_mat = torch.nn.Linear(
                    sum(self.density_n_comp), self.density_dim, bias=False
                ).to(device)
                self.app_basis_mat = torch.nn.Linear(
                    sum(self.app_n_comp), self.app_dim, bias=False
                ).to(device)
        else:
            self.density_basis_mat = torch.nn.Linear(
                self.density_n_comp[0], self.density_dim, bias=False
            ).to(device)
            self.app_basis_mat = torch.nn.Linear(
                self.app_n_comp[0], self.app_dim, bias=False
            ).to(device)

        # Initialize the basis matrices
        with torch.no_grad():
            weights = torch.ones_like(self.density_basis_mat.weight) / float(
                self.density_dim
            )
            self.density_basis_mat.weight.copy_(weights)

    def init_one_hexplane(self, n_component, gridSize, device):
        plane_coef, line_time_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn((1, n_component[i], gridSize[vec_id], self.time_grid))
                    + self.init_shift
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_time_coef
        ).to(device)

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.density_line_time,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.density_plane,
                "lr": lr_scale * cfg.lr_density_grid,
                "lr_org": cfg.lr_density_grid,
            },
            {
                "params": self.app_line_time,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.app_plane,
                "lr": lr_scale * cfg.lr_app_grid,
                "lr_org": cfg.lr_app_grid,
            },
            {
                "params": self.density_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_density_nn,
                "lr_org": cfg.lr_density_nn,
            },
            {
                "params": self.app_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_app_nn,
                "lr_org": cfg.lr_app_nn,
            },
        ]

        if isinstance(self.app_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.app_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_app_nn,
                    "lr_org": cfg.lr_app_nn,
                }
            ]

        if isinstance(self.density_regressor, torch.nn.Module):
            grad_vars += [
                {
                    "params": self.density_regressor.parameters(),
                    "lr": lr_scale * cfg.lr_density_nn,
                    "lr_org": cfg.lr_density_nn,
                }
            ]

        if self.use_codec:
            for idx in range(self.num_planes):
                codec_params = self._get_codec_params(self.app_nets[idx])
                grad_vars.append(
                    {
                        "params": codec_params['net'],
                        "lr": lr_scale * cfg.lr_codec_net,
                        "lr_org": cfg.lr_codec_net,
                    }
                )
                grad_vars.append(
                    {
                        "params": codec_params['aux'],
                        "lr": lr_scale * cfg.lr_codec_aux,
                        "lr_org": cfg.lr_codec_aux,
                    }
                )

        return grad_vars

    def _get_codec_params(self, net):
        params = {
            'net': [],
            'aux': []
        }
        for name, param in net.named_parameters():
            if param.requires_grad:
                if name.endswith('.quantiles'):
                    params['aux'].append(param)
                else:
                    params['net'].append(param)
        return params['net'], params['aux']

    def compute_densityfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Compuate the density features of sampled points from density HexPlane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            density: (N) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat = [], []
        # Extract features from six feature planes.
        for idx_plane in range(len(self.density_plane)):
            # Spatial Plane Feature: Grid sampling on density plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.density_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on density line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.density_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_feat, line_time_feat = torch.stack(plane_feat, dim=0), torch.stack(
            line_time_feat, dim=0
        )

        # Fusion One
        if self.fusion_one == "multiply":
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.density_basis_mat(inter.T)  # Feature Projection

        return inter

    def compute_appfeature(
        self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Compuate the app features of sampled points from appearance HexPlane.

        Args:
            xyz_sampled: (N, 3) sampled points' xyz coordinates.
            frame_time: (N, 1) sampled points' frame time.

        Returns:
            app_feature: (N, self.app_dim) density of sampled points.
        """
        # Prepare coordinates for grid sampling.
        # plane_coord: (3, B, 1, 2), coordinates for spatial planes, where plane_coord[:, 0, 0, :] = [[x, y], [x,z], [y,z]].
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.app_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.app_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.app_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat, line_time_feat = torch.stack(plane_feat), torch.stack(
            line_time_feat
        )

        # Fusion One
        if self.fusion_one == "multiply":
            inter = plane_feat * line_time_feat
        elif self.fusion_one == "sum":
            inter = plane_feat + line_time_feat
        elif self.fusion_one == "concat":
            inter = torch.cat([plane_feat, line_time_feat], dim=0)
        else:
            raise NotImplementedError("no such fusion type")

        # Fusion Two
        if self.fusion_two == "multiply":
            inter = torch.prod(inter, dim=0)
        elif self.fusion_two == "sum":
            inter = torch.sum(inter, dim=0)
        elif self.fusion_two == "concat":
            inter = inter.view(-1, inter.shape[-1])
        else:
            raise NotImplementedError("no such fusion type")

        inter = self.app_basis_mat(inter.T)  # Feature Projection

        return inter

    def TV_loss_density(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx]) + reg2(self.density_line_time[idx])
            )
        return total

    def TV_loss_app(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) + reg2(self.app_line_time[idx])
        return total

    def L1_loss_density(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line_time[idx]))
            )
        return total

    def L1_loss_app(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.app_plane[idx]))
                + torch.mean(torch.abs(self.app_line_time[idx]))
            )
        return total

    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, line_time_coef, res_target, time_grid):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
            line_time_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_time_coef[i].data,
                    size=(res_target[vec_id], time_grid),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )

        return plane_coef, line_time_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.app_plane, self.app_line_time = self.up_sampling_planes(
            self.app_plane, self.app_line_time, res_target, time_grid
        )
        self.density_plane, self.density_line_time = self.up_sampling_planes(
            self.density_plane, self.density_line_time, res_target, time_grid
        )

        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")

    def save_compressed(self, save_path):
        """
        Compresses the model parameters and saves the compressed data.

        Args:
            save_path (str): Path to save the compressed file.
        """
        compressed_data = {
            "app_plane": {name: param.cpu() for name, param in self.app_plane.named_parameters()},
            "app_rate": self.app_rate if hasattr(self, 'app_rate') else None,
            # Add other compressed components if necessary
        }
        torch.save(compressed_data, save_path)

    def load_compressed(self, compressed_path, device):
        """
        Loads the compressed data into the model.

        Args:
            compressed_path (str): Path to the compressed file.
            device (torch.device): Device to load the parameters on.
        """
        compressed_data = torch.load(compressed_path, map_location=device)
        self.app_plane.load_state_dict(compressed_data["app_plane"])
        if "app_rate" in compressed_data and compressed_data["app_rate"] is not None:
            self.app_rate = compressed_data["app_rate"]
        # Load other compressed components if necessary
