#!/usr/bin/env python3

# Copyright (C) 2025  Xiaofeng Yan
# Xueming Li Lab, Tsinghua University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.fft
import torch.cuda
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import opt_einsum as oe
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
from time import time
import json
import mrcfile

from mpicker_class2d_utils import Array1D, Array2D, Array3D, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor2D_c, Tensor3D_c, Tensor4D_c
from mpicker_class2d_utils import Mode, Star, Mask2D, Rotation2D
from mpicker_class2d_utils import read_star_loop, read_star_list, write_star_loop, write_star_list, get_mrc_idx_name, get_mrc_2d, write_mrcs, get_free_port
from mpicker_class2d_utils import rfft2_center, irfft2_center, rfft_radius, crop2D, pad2D, get_yx_for_pad, sinc, kaiser_Fourier_value
from mpicker_class2d_utils import replace_nan_inf, get_res_from_ssnr, fix_sigma2_noise, random_perturb, sample_angles, sample_offsets
from mpicker_class2d_utils import shift_img_int, get_ctf_image, remove_small_weight, argmax_weight, back_radial_average, imgs_phase_random


class Global:
    """Some global settings"""
    find_GroupId = True
    group_key_order = [Star.OpticsGroup, Star.GroupName, Star.MicrographName]
    batch_init = 256
    batch_exp = 64
    init_group_same = True # when init, if max_init>0, let all groups have same sigma2noise

    # something different from Relion but may be important for tomo projections with missing wedge:
    ctf_mask_thres = 1e-3 # the region of images where abs(ctf)<=thres will be masked (2D missing wedge). <0 to close. not sure is it useful
    max_current_size = -50 # <=0 to close. only use low resolution.
    use_entropy = 0 # from this iteration (<1 to close), weight particles by entropy when reconstruction (0.1 to 1.0)
    output_entropy = True # output 2^entropy as rlnNrOfSignificantSamples in _data.star (the definition is different from Relion)
    adjust_edge = True # auto increase edge size for large image
    edge_real = None # if not None, use this value as edge_5. using large edge_real (soft) may be useful
    skip_particle_mask = False # just skip masking particles (only masking refs)
    noise_particle_mask = False # fill the region out of mask with phase randomized noise (for particles)
    skip_ref_mask = False # skip masking refs after reconstruction. if particles were masked, masking refs may be unnecessary
    skip_ref_mask_E = True # skip masking refs after rotation in E step, seems unnecessary?

    output_probability = False # output class probability in _prob.txt
    probability_max = True # when output_probability, output max or sum for each class

    batch_worker = 1 # 0 to close. 1 seems enough
    pin_memory = False # seems not so useful
    size_ratio_init = 0.14
    edge_2 = 2
    edge_3 = 3
    edge_5 = 5
    edge_7 = 7
    weight_niter = 10 # <=0 to skip it, Relion is 10
    weight_small = 1e-4
    sigma_offset = 3.
    incr_size = 10
    purturb = True
    skip_align = False
    gridding_before_recon = True # Relion is False
    shift_wrap = True # Relion is False
    prior_classs = False # Relion is only True when skipalign
    prior_offset = True
    adaptive_fraction = 0.999 + 1 # >=1 to skip it. weight are not sparse for me...
    scale_ssnr_thres = 3. # as Relion, only use region with ssnr>3 when scale correction
    minres_map = 5 # as Relion, ignore tau2 in low frequency(<5) when reconstrution
    scale_range_thres = 5 # as Relion, clamp between 1/5*median and 5*median when scale correction
    mask_before_pad_f = True # not sure is it useful
    mask_before_pad_r = False # seems not a good try
    mpi_average = False # after M step, just use result of rank0 or average all results
    mpi_verbose = False # print when each rank merge


def set_seed(seed: int) -> None:
    """Set random seed for numpy and torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.set_deterministic(True)


def gridding_correction(imgs: Tensor3D, full_size: float, masker: Mask2D) -> Tensor3D:
    """full_size and masker can be None\n
       full_size means there might be padding\n
       masker should be img_size real if provided"""
    img_size = imgs.size(-2)
    if full_size is None:
        full_size = img_size
    if masker is None:
        masker = Mask2D(img_size, is_rfft=False, device=imgs.device)
    sinc2 = sinc(masker.get_grid_dist() / full_size) ** 2
    sinc2[sinc2 < 1e-2] = 1e-2
    result = imgs / sinc2[None,:,:]
    return result

def gridding_correction_rfft(imgs: Tensor3D, full_size: float, masker: Mask2D) -> Tensor3D:
    """pad to full_size, irfft, gridding, rfft, crop\n
       masker can be None\n
       masker should be full_size real if provided"""
    img_size = imgs.size(-2)
    imgs = pad2D(imgs, full_size, is_rfft=True, fill=Mode.pad_zero)
    imgs = irfft2_center(imgs)
    imgs = gridding_correction(imgs, full_size, masker)
    imgs = rfft2_center(imgs)
    imgs = crop2D(imgs, img_size, is_rfft=True)
    return imgs


def data_over_weight(data: Tensor3D_c, weight: Tensor3D, niter: int, radius:float=None) -> Tensor3D_c:
    """will change small value in weight in place if niter<=0\n
       radius can be radius(current_size)*pad?"""
    img_size = data.size(-2)
    if radius is None:
        radius = rfft_radius(img_size) # because padding, radius may samll than it?
    masker = Mask2D(img_size, is_rfft=True, device=data.device)
    masker.set_soft_mask(radius-Global.edge_3+1, edge=Global.edge_3, default_bgmode=Mode.mask_zero)

    if niter <= 0:
        weight[weight < Global.weight_small] = Global.weight_small
        data[weight < Global.weight_small] = 0
        result = data / weight
        return masker.apply_mask(result)
    
    masker_real = Mask2D(img_size, is_rfft=False, device=data.device)
    k_f_window = kaiser_Fourier_value(masker_real.get_grid_dist() / (radius*2)) # /img_size may cause problem??
    inv_weight = torch.ones_like(weight)
    weight = masker.apply_mask(weight)
    for i in range(niter):
        inv_weight_weight = inv_weight * weight
        # convolve with blob
        inv_weight_weight = irfft2_center(inv_weight_weight)
        inv_weight_weight *= k_f_window[None,:,:]
        inv_weight_weight = rfft2_center(inv_weight_weight).abs()
        # convolve end
        mask = inv_weight_weight < Global.weight_small
        inv_weight_weight[mask] = Global.weight_small
        inv_weight /= inv_weight_weight
        inv_weight = replace_nan_inf(inv_weight, 0.)
    masker.set_soft_mask(radius-Global.edge_3, edge=Global.edge_3, default_bgmode=Mode.mask_zero) # ignore the most outer circle
    inv_weight = masker.apply_mask(inv_weight)
    result = data * inv_weight
    return result


class DatasetInitial(Dataset):
    def __init__(self, data_dicts: List[Dict[str, str]], group_ids: List[int], pixel_size: float=None,
                 img_dict:dict=None, ctf_dict:dict=None):
        """data_dicts: the result of read_star_loop.\n
           Will generate img, ctf, group_id."""
        self.data_dicts: List[Dict[str, str]] = data_dicts
        if len(data_dicts) == 0: return
        self.has_rlnCtfImage: bool = Star.CtfImage in data_dicts[0].keys()
        self.group_ids: Array1D = np.array(group_ids, dtype=int)
        img = get_mrc_2d(data_dicts[0][Star.ImageName])
        self.size: int = img.shape[0]
        self.prepare_ctf_conjugation()
        self.pixel_size: float = pixel_size
        has_defocus = Star.DefocusU in data_dicts[0].keys()
        self.generate_ctf = False
        if not self.has_rlnCtfImage and has_defocus and self.pixel_size is not None:
            self.generate_ctf = True
            self.prepare_ctf_generation()
        self.img_mrcs = img_dict
        self.ctf_mrcs = ctf_dict

    def __len__(self):
        return len(self.data_dicts)
    
    def prepare_ctf_conjugation(self):
        # just for correction of negtive half y axis
        ypos = self.size//2 + 1
        yneg = self.size//2 - 1
        ylen = (self.size - 1)//2
        self.startn: int = yneg
        self.endn: int = yneg-ylen
        self.startp: int = ypos
        self.endp: int = ypos+ylen

    def prepare_ctf_generation(self):
        # will calculate ctf using defocus
        masker = Mask2D(self.size, is_rfft=True, device="cpu")
        dist_ctf = masker.get_grid_dist().numpy()
        length = self.size*self.pixel_size
        self.u2: Array2D = (dist_ctf/length)**2
        self.u4: Array2D = self.u2**2
        gridy, gridx = masker.get_grid_yx().numpy()
        ang = np.arctan2(gridy, gridx)
        self.sin_2: Array2D = np.sin(2*ang)
        self.cos_2: Array2D = np.cos(2*ang)

    def get_mrc_from_dict(self, pname:str, dict_mrc:dict) -> Array2D:
        idx, fname = get_mrc_idx_name(pname)
        if idx is None:
            return dict_mrc[fname]
        else:
            return dict_mrc[fname][idx]

    def __getitem__(self, idx) -> Tuple[Array2D, Array2D, int]:
        # particle
        pname = self.data_dicts[idx][Star.ImageName]
        if self.img_mrcs is not None:
            img = self.get_mrc_from_dict(pname, self.img_mrcs)
        else:
            img = get_mrc_2d(pname)
        # ctf
        if self.has_rlnCtfImage:
            pname = self.data_dicts[idx][Star.CtfImage]
            if self.ctf_mrcs is not None:
                ctf = self.get_mrc_from_dict(pname, self.ctf_mrcs)
            else:
                ctf = get_mrc_2d(pname)
        elif self.generate_ctf:
            ctf = get_ctf_image(self.u2, self.u4, self.sin_2, self.cos_2, self.data_dicts[idx])
        else:
            ctf = np.ones_like(img)
        if ctf.shape[0] == ctf.shape[1]:
            ctf = np.roll(ctf, -1, 1)[:,self.size//2-1:]
        ctf[self.startn:self.endn:-1, 0] = ctf[self.startp:self.endp, 0] # conjugation
        return img, ctf, self.group_ids[idx]
    

class DatasetExpectation(DatasetInitial):
    def __init__(self, data_dicts: List[Dict[str, str]], group_ids: List[int], norms: List[float], shifty: List[float], shiftx: List[float], wrap: bool, pixel_size: float=None,
                 img_dict:dict=None, ctf_dict:dict=None):
        """Will generate img, ctf, group_id, norm, and rounded shifty, shiftx.\n
           if not wrap, will fill 0 when translate."""
        super().__init__(data_dicts, group_ids, pixel_size, img_dict, ctf_dict)
        if len(data_dicts) == 0: return
        self.norms: Array1D = np.array(norms)
        self.shifty: Array1D = (np.array(shifty) + 0.5).astype(int)
        self.shiftx: Array1D = (np.array(shiftx) + 0.5).astype(int)
        self.wrap: bool = wrap

    def __getitem__(self, idx) -> Tuple[Array2D, Array2D, int, float, int, int]:
        img, ctf, group_id = super().__getitem__(idx)
        norm = self.norms[idx]
        sy = self.shifty[idx]
        sx = self.shiftx[idx]
        if self.wrap:
            img = np.roll(img, (sy, sx), axis=(0, 1))
        else:
            img = shift_img_int(img, sy, sx, 0) # fill 0 as Relion
        return img, ctf, group_id, norm, sy, sx


class Class2D:
    def __init__(self, data_dicts: List[Dict[str, str]], num_class: int, num_iter: int, out_pre: str,
                 device: torch.device="cpu", random_ang: Array1D=None, random_x: Array1D=None, random_y: Array1D=None,
                 pixel_size: float=None, diameter: float=None, 
                 psi_step=6., offset_step=1., offset_range=5., T=2., pad=2., 
                 init_max=0, normalize=False, Cn=1, img_ctf_dict:Tuple[dict]=None,
                 rank=0):
        self.rank:int = rank # only for mpi
        # for inputs
        self.data_dicts: List[Dict[str, str]] = data_dicts # will update
        self.data_keys: List[str] = list(data_dicts[0].keys()) # might change later
        self.num_class: int = num_class
        self.num_iter: int = num_iter
        self.out_pre: str = out_pre
        self.device: torch.device = device
        self.psi_step: float = psi_step
        self.offset_step: float = offset_step
        self.offset_range: float = offset_range
        self.pixel_size: float = pixel_size # in Angstrom
        self.diameter: float = diameter # in Angstrom
        self.radius: int # in pixel
        self.ori_size: int # image size
        self.ori_size_pad: int
        self.pad: float = pad
        self.init_max: int = init_max if len(data_dicts) > init_max else 0 # use a few particles for initialization to speed up
        self.normalize: bool = normalize
        self.Cn: int = Cn
        self.T: float = T
        self.rmax_ori: int
        self.get_basic_info()
        self.group_ids: List[int]
        self.num_group: int
        self.group_names: List[str]
        self.group_key: str = Star.GroupName
        self.get_GroupId(Global.group_key_order)

        # random sampling
        if random_ang is None:
            self.random_ang: Array1D = random_perturb(self.num_iter) # -0.5 to 0.5
        else:
            self.random_ang: Array1D = random_ang
        if random_x is None:
            self.random_x: Array1D = random_perturb(self.num_iter)
        else:
            self.random_x: Array1D = random_x
        if random_y is None:
            self.random_y: Array1D = random_perturb(self.num_iter)
        else:
            self.random_y: Array1D = random_y
        if Global.skip_align or not Global.purturb:
            self.random_ang.fill(0)
            self.random_x.fill(0)
            self.random_y.fill(0)
        self.sample_ang: Tensor1D
        self.sample_x: Tensor1D
        self.sample_y: Tensor1D
        self.offset_images: Tensor3D_c # one image for each offset, current_size

        # model data
        self.sigma_offset: float = Global.sigma_offset # will update
        self.sigma2_noise: Tensor2D
        self.tau2_class_T: Tensor2D # after multiply T
        self.tau2_class_T_old: Tensor2D # recorded tau is from the ref of last iter
        self.sigma2_class: Tensor2D
        self.ssnr: Tensor2D
        self.current_resolutions = Tensor1D
        self.norms_mean: float
        self.pdf_class: Tensor1D

        # data
        self.refs: Tensor3D 
        """real space, ori_size"""
        self.class_ids_s1: List[int] # start from 1
        self.orignXs: List[float]
        self.orignYs: List[float]
        self.angles_deg: List[float] # in degrees
        self.norms: List[float]
        self.scales_group: Tensor1D
        if Global.output_entropy:
            self.entropys: List[float]
        if Global.output_probability:
            self.class_prob: List[List[float]]
        # can load all mrcs files into memory to speed up
        if img_ctf_dict is not None:
            self.img_dict: Dict[str, Tensor3D] = img_ctf_dict[0]
            self.ctf_dict: Dict[str, Tensor3D] = img_ctf_dict[1]
        else:
            self.img_dict = None
            self.ctf_dict = None

        # for calculation
        self.niter = 0 # current iter, as Relion, start from 1. 0 means init
        self.current_size: int
        self.current_size_pad: int
        self.current_pad: float # same as pad if it is int
        self.current_padnorm: float # we pad in real space and use norm=forward in fft as Relion
        self.rmax_cur: int
        self.rmax_cur_pad: int
        self.masker_real: Mask2D
        """ori_size, real, radius edge_5, bgmean"""
        self.masker_pad: Mask2D
        """ori_size, real, edge_3, bgmean. apply before pad"""
        self.masker_rfft: Mask2D
        """ori_size, rfft, no mask. for radial average"""
        self.get_basic_mask()
        self.masker_recon: Mask2D # update in updata_current_size()
        """current_size_pad, rfft, edge_3, bgzero"""
        self.masker_sigma: Mask2D
        """current_size, rfft, 0, bgzero"""
        self.masker_current_real: Mask2D
        """current_size, real, edge_3, bgzero"""
        self.masker_current_fft: Mask2D
        """current_size, rfft, edge_7, bgzero"""
        self.masker_invtau: Mask2D
        """current_size_pad, rfft, edge_7*2, bgzero"""
        self.sum_weight: float
        self.sum_norms: float
        self.sum_groupnum: Tensor1D
        self.sum_classweight: Tensor1D
        self.refs_rotated: Tensor4D_c
        """fourier space, current_size. class*angle*xxx"""
        self.invSigma2: Tensor3D
        self.mask_small_ssnr: Tensor3D # bool
        self.size_big_ssnr: int
        # for calculation, might be cleared
        self.sum_init_fft: Tensor2D_c
        self.sum_init_fft2_group: Tensor3D
        self.wsum_diff2_group: Tensor3D
        self.wsum_data_unrot: Tensor4D_c # correspond to refs_rotated
        self.wsum_weight_unrot: Tensor4D
        self.wsum_data: Tensor3D_c
        self.wsum_weight: Tensor3D
        self.wsum_XA_group: Tensor1D
        self.wsum_AA_group: Tensor1D
        self.wsum_offset_diff2: float

    def get_basic_info(self):
        """get ori_size and pixel_size from data_dicts, and radius pad"""
        if Star.ImageName not in self.data_keys:
            raise Exception(f"{Star.ImageName} not in data keys")
        dict0 = self.data_dicts[0]
        img = get_mrc_2d(dict0[Star.ImageName])
        sy, sx = img.shape
        if sy != sx:
            raise Exception(f"Image shape is not square: {sx} x {sy}")
        self.ori_size = sx
        if Global.adjust_edge:
            multi_edge = int(self.ori_size / 64 + 0.5)
            Global.edge_2 = 2 * multi_edge
            Global.edge_3 = 3 * multi_edge
            Global.edge_5 = 5 * multi_edge
            Global.edge_7 = 7 * multi_edge
        if Global.edge_real is not None:
            Global.edge_5 = Global.edge_real
        if self.pixel_size is None:
            if Star.Magnification not in self.data_keys or Star.DetectorPixelSize not in self.data_keys:
                raise Exception(f"Not found {Star.Magnification} {Star.DetectorPixelSize}, provide pixel_size manually")
            self.pixel_size = 10000 * float(dict0[Star.DetectorPixelSize]) / float(dict0[Star.Magnification])
            if self.rank==0: print(f"Use pixel_size = {self.pixel_size:.2f} A")
        if self.diameter is None:
            self.diameter = self.ori_size * self.pixel_size
        self.radius = int(self.diameter / self.pixel_size / 2)
        self.diameter = self.radius * self.pixel_size * 2
        if self.rank==0: print(f"Use diameter = {self.diameter:.2f} A, radius = {self.radius} pixel")
        self.ori_size_pad = int(self.ori_size * self.pad + 0.5)
        self.rmax_ori = rfft_radius(self.ori_size)

    def get_basic_mask(self):
        self.masker_real = Mask2D(self.ori_size, is_rfft=False, device=self.device) # mask particle
        self.masker_real.set_soft_mask(self.radius, Global.edge_5, Mode.mask_bgmean)
        self.masker_pad = Mask2D(self.ori_size, is_rfft=False, device=self.device) # for pad in real space
        self.masker_pad.set_soft_mask(self.rmax_ori - Global.edge_3, Global.edge_3, Mode.mask_bgmean)
        self.masker_rfft = Mask2D(self.ori_size, is_rfft=True, device=self.device) # can used for radial average, lowpass

    def get_GroupId(self, group_key_order: List[str]):
        """get group_ids from data_dicts based on GroupName or MicrographName"""
        find_GroupId = False
        for key in group_key_order:
            if key in self.data_keys:
                self.group_key = key # used when write _model.star
                find_GroupId = True
                if self.rank==0: print(f"Use {key} for group id")
                break
        if not find_GroupId:
            self.num_group = 1
            self.group_ids = [0] * len(self.data_dicts)
            self.group_names = ["none"]
            if self.rank==0: print(f"Set all group ids to 0")
            return
        
        self.num_group = 0
        self.group_ids = []
        self.group_names = []
        name2id = dict()
        for data in self.data_dicts:
            group_name = data[self.group_key]
            if group_name not in name2id.keys():
                self.group_names.append(group_name)
                name2id[group_name] = self.num_group
                self.num_group += 1 
            self.group_ids.append(name2id[group_name])

    def get_resolutions(self) -> Tuple[Tensor1D, Tensor1D, Tensor1D]:
        """return idx(from 0), Fourier resolution, and real resolution\n
           the first resolution is set to 999 A\n"""
        idx = torch.arange(self.rmax_ori+1, device=self.device, dtype=torch.int32)
        res_f = idx / (self.pixel_size * self.ori_size)
        res_f[0] = 1/999
        res_r = 1/res_f
        res_f[0] = 0
        return idx, res_f, res_r
    
    def get_invSigma2(self):
        """get invsigma2 for each group and save as self.invSigma2\n
           ignore the origin point, negative half y axis, and region out of radius\n
           also find mask of small ssnr and save as self.mask_small_ssnr\n"""
        idx = self.masker_sigma.get_grid_dist()
        invsigma2 = back_radial_average(self.sigma2_noise, idx, interp=True)
        invsigma2 = 1/invsigma2
        invsigma2 = replace_nan_inf(invsigma2, 0.) # in gerenal won't happen
        self.invSigma2 = invsigma2
        # when scale correction, conly consider big ssnr
        ssnr = back_radial_average(self.ssnr, idx, interp=True)
        mask_small_ssnr = ssnr <= Global.scale_ssnr_thres
        # # size of big ssnr, to speedup
        if mask_small_ssnr.all():
            radius = 0
        else:
            radius = int(idx.expand(len(ssnr),-1,-1)[~mask_small_ssnr].max()+0.5)
        self.size_big_ssnr = 2 * int(radius+2)
        if self.size_big_ssnr > self.current_size:
            self.size_big_ssnr = self.current_size
        self.mask_small_ssnr = crop2D(mask_small_ssnr, self.size_big_ssnr, is_rfft=True)
        if self.rank==0: print(f"Iter{self.niter:03d}, Current size = {self.current_size}, Big ssnr size = {self.size_big_ssnr}")

    def wiener_filter_modification(self):
        """will monify self.wsum_weight (add invtau2)"""
        invtau2 = back_radial_average(self.tau2_class_T, self.masker_recon.get_grid_dist()/self.current_pad, interp=True)
        invtau2 = 1/invtau2
        if Global.mask_before_pad_f:
            invtau2 = self.masker_invtau.apply_mask(invtau2)
        mask = ~torch.isfinite(invtau2)
        invtau2[mask] = self.wsum_weight[mask] * 1000 # as Relion
        mask = self.masker_recon.get_grid_dist() < Global.minres_map
        invtau2[:, mask] = 0
        self.wsum_weight += invtau2

    def get_offset_images(self):
        """get a complex image for each offset and save as self.offset_images\n
           for the image shift on the fly. current_size."""
        gridy, gridx = self.masker_sigma.get_grid_yx() # current_size
        phase = self.sample_y[:,None,None]*gridy[None,:,:] + self.sample_x[:,None,None]*gridx[None,:,:]
        phase *= -2*np.pi/self.ori_size
        self.offset_images = torch.exp(1j*phase)

    def update_sampling(self, niter: int):
        pa, px, py = self.random_ang[niter-1], self.random_x[niter-1], self.random_y[niter-1]
        self.sample_ang = sample_angles(self.psi_step, pa, Cn=self.Cn).to(self.device)
        self.sample_x, self.sample_y = sample_offsets(self.offset_step, self.offset_range, px, py).to(self.device)
    
    def updata_current_size(self, current_size: int):
        """update self.current_size, mask, and so on"""
        if current_size > self.ori_size:
            current_size = self.ori_size
        if Global.max_current_size > 0 and current_size > Global.max_current_size:
            current_size = Global.max_current_size
        self.current_size = current_size
        self.current_size_pad = int(self.current_size * self.pad + 0.5)
        self.current_pad = self.current_size_pad / self.current_size
        self.current_padnorm = self.current_pad**2
        self.rmax_cur = rfft_radius(self.current_size)
        self.rmax_cur_pad = int(rfft_radius(self.current_size)*self.current_pad)

        self.masker_recon = Mask2D(self.current_size_pad, is_rfft=True, device=self.device)
        self.masker_recon.set_soft_mask(self.rmax_cur_pad-Global.edge_3, Global.edge_3, Mode.mask_zero)
        self.masker_sigma = Mask2D(self.current_size, is_rfft=True, device=self.device)
        self.masker_sigma.set_soft_mask(self.rmax_cur, 0, Mode.mask_zero)
        # pad before backrot, only for Global.mask_before_pad
        self.masker_current_real = Mask2D(self.current_size, is_rfft=False, device=self.device)
        self.masker_current_real.set_soft_mask(self.rmax_cur-Global.edge_3, Global.edge_3, Mode.mask_zero)
        r, edge = self.rmax_cur-Global.edge_7, Global.edge_7 # mask_before_pad, soft mask may reduce the artifact of padding?
        pad_r, pad_edge = r*self.current_pad, edge*self.current_pad
        self.masker_current_fft = Mask2D(self.current_size, is_rfft=True, device=self.device) 
        self.masker_current_fft.set_soft_mask(r, edge, Mode.mask_zero)
        self.masker_invtau = Mask2D(self.current_size_pad, is_rfft=True, device=self.device)
        self.masker_invtau.set_soft_mask(pad_r, pad_edge, Mode.mask_zero) # data weight invtau2 should masked together

    def data_vs_prior_class(self, iter0=False):
        """calculate ssnr and so on"""
        # myssnr=evidence/prior=tau2/sigma2, evidence = weight = 1/sigma2, prior = 1/tau2
        # maybe the radial average of (spectrum*weight) is another choice
        ssnr = self.tau2_class_T / self.sigma2_class
        if not iter0:
            ssnr[:, self.rmax_cur+1:] = 0 # Relion skip it when iter0
        self.ssnr = replace_nan_inf(ssnr, 0.)
        del ssnr
        # current_resolutions. 1/res = n * 1/img_size
        _, _, resolutions = self.get_resolutions()
        self.current_resolutions = get_res_from_ssnr(self.ssnr, resolutions)

    def no_missing_wedge_average(self, mask: Tensor3D) -> Tensor3D:
        """mask is ctf<=thres. will output radial average images"""
        noMW_ra = 1 - self.masker_rfft.radial_average(mask.to(torch.float32), self.rmax_ori)
        noMW_ra[noMW_ra<0.1] = 0.1
        noMW_ra = F.max_pool1d(noMW_ra.unsqueeze(0), kernel_size=3, padding=1, stride=1).squeeze(0) # smooth
        noMW_ra: Tensor3D = back_radial_average(noMW_ra, self.masker_rfft.get_grid_dist(), interp=True)
        return noMW_ra

    def reconstruction(self, iter0=False):
        """use wiener_filter and normalize wsum_data and wsum_weight, to get refs"""
        if Global.gridding_before_recon:
            # correction for interpolation
            self.wsum_data = gridding_correction_rfft(self.wsum_data, self.ori_size_pad, None)
            self.wsum_weight = gridding_correction_rfft(self.wsum_weight, self.ori_size_pad, None).abs()
        if not iter0:
            self.wiener_filter_modification()
        # normalize as Relion. make the value comparable to Global.weight_small?
        self.wsum_data /= self.sum_classweight[:,None,None]
        self.wsum_weight /= self.sum_classweight[:,None,None]
        self.refs = data_over_weight(self.wsum_data, self.wsum_weight, Global.weight_niter, self.rmax_cur_pad)
        # its value should correspond to the padded image, but now the value is from unpadded iamge, so we need to correct it
        self.refs /= self.current_padnorm
        self.refs = self.masker_recon.apply_mask(self.refs)
        self.refs = pad2D(self.refs, self.ori_size_pad, is_rfft=True, fill=Mode.pad_zero)
        self.refs = irfft2_center(self.refs)
        self.refs = crop2D(self.refs, self.ori_size, is_rfft=False)
        if not Global.gridding_before_recon:
            self.refs = gridding_correction(self.refs, self.ori_size_pad, self.masker_real) # as Relion
    
    def initialize_batch(self, imgs: Tensor3D, ctfs: Tensor3D, groupIds: Tensor1D):
        if Global.ctf_mask_thres >= 0:
            mask = ctfs.abs() <= Global.ctf_mask_thres
            noMW_ra = self.no_missing_wedge_average(mask)

        length = len(imgs)

        if Global.noise_particle_mask:
            backgrounds = imgs_phase_random(imgs)
            imgs = self.masker_real.apply_mask(imgs, bgs=backgrounds)
        else:
            imgs = self.masker_real.apply_mask(imgs) # bgmean

        # for initail sigma2_noise
        imgs_rfft = rfft2_center(imgs)
        if Global.ctf_mask_thres >= 0:
            imgs_rfft[mask] = 0.
            imgs_rfft /= noMW_ra**0.5
        imgs_rfft2 = imgs_rfft.real**2 + imgs_rfft.imag**2
        self.sum_init_fft += imgs_rfft.sum(dim=0)
        self.sum_init_fft2_group.index_add_(0, groupIds, imgs_rfft2)
        del imgs_rfft, imgs_rfft2

        # for initail ref
        # pad imgs
        imgs = self.masker_pad.apply_mask(imgs)
        imgs = pad2D(imgs, self.ori_size_pad, is_rfft=False)
        imgs_rfft = rfft2_center(imgs)
        imgs_rfft = crop2D(imgs_rfft, self.current_size_pad, is_rfft=True)
        imgs_rfft *= self.current_padnorm
        del imgs
        # pad ctf
        ictfs = irfft2_center(ctfs)
        ictfs = self.masker_pad.apply_mask(ictfs, Mode.mask_zero)
        ictfs = pad2D(ictfs, self.ori_size_pad, is_rfft=False)
        ctfs = rfft2_center(ictfs).real
        ctfs = crop2D(ctfs, self.current_size_pad, is_rfft=True)
        ctfs *= self.current_padnorm
        del ictfs
        if Global.ctf_mask_thres >= 0:
            mask = ctfs.abs() <= Global.ctf_mask_thres
            imgs_rfft[mask] = 0.
            ctfs[mask] = 0.
        # rotate
        angles = torch.rand(length, device=self.device) * (2*np.pi) - np.pi
        if Global.skip_align: angles.fill_(0.)
        classIds = torch.randint(self.num_class, (length,), device=self.device)
        rotater = Rotation2D(angles, self.current_size_pad, is_rfft=True, direction=Mode.anti_clockwise)
        data = rotater.rotate(imgs_rfft*ctfs, Mode.rot_N_N, complex=True)
        self.wsum_data.real.index_add_(0, classIds, data.real)
        self.wsum_data.imag.index_add_(0, classIds, data.imag)
        del imgs_rfft, data
        weight = rotater.rotate(ctfs*ctfs, Mode.rot_N_N, complex=False)
        self.wsum_weight.index_add_(0, classIds, weight)
        del ctfs, weight

        # count
        self.sum_weight += length
        self.sum_groupnum.put_(groupIds, torch.ones_like(groupIds), accumulate=True)
        self.sum_classweight.put_(classIds, torch.ones_like(classIds, dtype=torch.float32), accumulate=True)
        self.sum_norms += length

        # update data
        self.class_ids_s1 += (classIds+1).tolist()
        self.angles_deg += torch.rad2deg(angles).tolist()
        self.orignXs += [0] * length
        self.orignYs += [0] * length
        self.norms += [1.] * length
        
    def initialize(self):
        current_size = int(self.ori_size * Global.size_ratio_init + 0.5)
        self.updata_current_size(current_size)

        self.class_ids_s1 = []
        self.orignXs = []
        self.orignYs = []
        self.angles_deg = []
        self.norms = []
        self.scales_group = torch.ones(self.num_group, device=self.device, dtype=torch.float32) # useless when initial
        
        self.sum_init_fft = torch.zeros(self.ori_size, self.ori_size//2+1, device=self.device, dtype=torch.complex64)
        self.sum_init_fft2_group = torch.zeros(self.num_group, self.ori_size, self.ori_size//2+1, device=self.device, dtype=torch.float32)
        self.wsum_data = torch.zeros(self.num_class, self.current_size_pad, self.current_size_pad//2+1, device=self.device, dtype=torch.complex64)
        self.wsum_weight = torch.zeros(self.num_class, self.current_size_pad, self.current_size_pad//2+1, device=self.device, dtype=torch.float32)
        self.sum_weight = 0.
        self.sum_norms = 0.
        self.sum_groupnum = torch.zeros(self.num_group, device=self.device, dtype=int)
        self.sum_classweight = torch.zeros(self.num_class, device=self.device, dtype=torch.float32)
        
        length = len(self.data_dicts)
        if self.init_max > 0 and self.init_max <= length:
            subset = np.random.permutation(length)[:self.init_max]
            data_dicts = [self.data_dicts[i] for i in subset]
            group_ids = [self.group_ids[i] for i in subset]
            dataset_init = DatasetInitial(data_dicts, group_ids, self.pixel_size, self.img_dict, self.ctf_dict)
        else:
            dataset_init = DatasetInitial(self.data_dicts, self.group_ids, self.pixel_size, self.img_dict, self.ctf_dict)
        dataloader_init = DataLoader(dataset_init, batch_size=Global.batch_init, num_workers=Global.batch_worker, 
                                     pin_memory=Global.pin_memory, shuffle=False, drop_last=False)
        if self.rank==0: pbar = tqdm(total=len(dataloader_init), desc="Init")
        for imgs, ctfs, groupIds in dataloader_init:
            imgs: Tensor3D = imgs.to(dtype=torch.float32, device=self.device)
            ctfs: Tensor3D = ctfs.to(dtype=torch.float32, device=self.device)
            groupIds: Tensor1D = groupIds.to(dtype=int, device=self.device)
            if self.normalize:
                imgs -= imgs.mean(dim=(1,2), keepdim=True)
                imgs /= imgs.std(dim=(1,2), keepdim=True) + 1e-6
            self.initialize_batch(imgs, ctfs, groupIds)
            if self.rank==0: pbar.update(1)
        if self.rank==0: pbar.close()

        if self.init_max > 0:
            # only part of particles are used, so just fill default values
            self.class_ids_s1 = [1] * length
            self.angles_deg = [0.] * length
            self.orignXs = [0.] * length
            self.orignYs = [0.] * length
            self.norms = [1.] * length
        if Global.output_entropy:
            self.entropys = [0.] * length

        self.mpi_initialize() # only for mpi

        self.rotate_wsum_Cn() # make wsum_data and wsum_weight satisfy Cn symmetry

        if self.init_max > 0 and Global.init_group_same:
            # let all groups have same initial values
            self.sum_init_fft2_group[:] = self.sum_init_fft2_group.mean(dim=0)
            self.sum_groupnum[:] = (self.sum_groupnum.to(torch.float32).mean(dim=0) + 0.5).to(int)

        # self.pdf_class = self.sum_classweight / self.sum_classweight.sum()
        self.pdf_class = torch.ones_like(self.sum_classweight) / self.num_class # set uniform when init
        self.norms_mean = self.sum_norms / self.sum_weight
        # sigma_noise
        mean_spectrum = self.sum_init_fft / self.sum_weight
        mean_spectrum = mean_spectrum.real**2 + mean_spectrum.imag**2
        spectrum_mean_group = self.sum_init_fft2_group / self.sum_groupnum[:,None,None]
        spectrum_mean_group -= mean_spectrum[None,:,:]
        self.sigma2_noise = self.masker_rfft.radial_average(spectrum_mean_group, self.rmax_ori) / 2
        self.sigma2_noise[:, 0] = self.sigma2_noise[:, 1] # avoid too small value for normalized (zero mean) images
        self.sigma2_noise = fix_sigma2_noise(self.sigma2_noise)
        del mean_spectrum, spectrum_mean_group
        self.sum_init_fft = None
        self.sum_init_fft2_group = None

        # reconstruction
        self.reconstruction(iter0=True)
        
        # lowpass, only when init
        self.refs = rfft2_center(self.refs)
        self.masker_rfft.set_soft_mask(self.rmax_cur, Global.edge_2, Mode.mask_zero)# for lowpass
        self.refs = self.masker_rfft.apply_mask(self.refs)
        self.refs = irfft2_center(self.refs)

        # add mask
        if not Global.skip_ref_mask:
            self.refs = self.masker_real.apply_mask(self.refs)

        # ssnr (data_vs_prior_class)
        sigma2_noise_mean = self.sigma2_noise * self.sum_groupnum[:,None]
        sigma2_noise_mean: Tensor1D = sigma2_noise_mean.sum(dim=0) / self.sum_groupnum.sum()
        ave_num = self.sum_weight/self.num_class
        self.sigma2_class = (sigma2_noise_mean/ave_num).repeat(self.num_class, 1) # an initial guess as Relion. 1 / (N * 1/sigma2)
        spectrum_class = rfft2_center(self.refs)
        spectrum_class = spectrum_class.real**2 + spectrum_class.imag**2
        self.tau2_class_T = self.masker_rfft.radial_average(spectrum_class, self.rmax_ori) / 2
        self.tau2_class_T[:, 0] = self.tau2_class_T[:, 1] # avoid too small value for normalized (zero mean) images
        self.tau2_class_T *= self.T
        self.tau2_class_T_old = self.tau2_class_T
        del spectrum_class, sigma2_noise_mean
        self.data_vs_prior_class(iter0=True)

        # write mrcs and star
        keys_add = [Star.ClassNumber, Star.AnglePsi, Star.OriginX, Star.OriginY, Star.NormCorrection]
        for k in keys_add:
            if k not in self.data_keys:
                self.data_keys.append(k)
        if Global.output_entropy and Star.NrOfSignificantSamples not in self.data_keys:
            self.data_keys.append(Star.NrOfSignificantSamples)
        self.update_data_dicts()
        self.write(0)

        # backup some data
        self.refs_init =self.refs.clone()
        self.tau2_class_T_init = self.tau2_class_T.clone()
        self.ssnr_init = self.ssnr.clone()

    def initialize_continue(self, fmodel:str):
        if self.rank==0: print(f"Loading {fmodel} to continue...")
        _, model_general = read_star_list(fmodel, "model_general")
        self.norms_mean = float(model_general[Star.NormCorrectionAverage])
        self.sigma_offset = float(model_general[Star.SigmaOffsets])
        current_size = int(model_general[Star.CurrentImageSize])
        self.updata_current_size(current_size)

        _, model_classes = read_star_loop(fmodel, "model_classes")
        
        pdf_class = [float(d[Star.ClassDistribution]) for d in model_classes]
        refs = [get_mrc_2d(d[Star.ReferenceImage]) for d in model_classes]
        current_resolutions = [float(d[Star.EstimatedResolution]) for d in model_classes]
        save_ref_id = [i for i in range(len(pdf_class)) if pdf_class[i]>=0] # can change ClassDistribution to negative to skip some classes
        refs = [refs[i] for i in save_ref_id]
        pdf_class = [pdf_class[i] for i in save_ref_id]
        current_resolutions = [current_resolutions[i] for i in save_ref_id]
        if len(save_ref_id) != self.num_class:
            if self.rank==0: print(f"{len(refs)} refs will be used, change num_class from {self.num_class} to {len(refs)}")
            self.num_class = len(refs)
        self.refs = torch.Tensor(refs).to(dtype=torch.float32, device=self.device)
        self.pdf_class = torch.Tensor(pdf_class).to(dtype=torch.float32, device=self.device)
        # current_size and current_resolutions will decide new current_size
        self.current_resolutions = torch.Tensor(current_resolutions).to(dtype=torch.float32, device=self.device)
        
        spectrum_class = rfft2_center(self.refs)
        spectrum_class = spectrum_class.real**2 + spectrum_class.imag**2
        self.tau2_class_T = self.masker_rfft.radial_average(spectrum_class, self.rmax_ori) / 2
        self.tau2_class_T[:, 0] = self.tau2_class_T[:, 1]
        self.tau2_class_T *= self.T # tau2_class_T will be used in next reconstruction

        ssnr = [] # ssnr will decide size_big_ssnr
        sigma2_class = []
        tau2_class = []
        for ic in save_ref_id:
            block = "model_class_" + str(ic+1)
            _, model_class = read_star_loop(fmodel, block)
            ssnr.append([float(d[Star.SsnrMap]) for d in model_class])
            sigma2_class.append([float(d[Star.ReferenceSigma2]) for d in model_class])
            tau2_class.append([float(d[Star.ReferenceTau2]) for d in model_class])
        self.ssnr = torch.Tensor(ssnr).to(dtype=torch.float32, device=self.device)
        self.sigma2_class = torch.Tensor(sigma2_class).to(dtype=torch.float32, device=self.device)
        self.tau2_class_T_old = self.T * torch.Tensor(tau2_class).to(dtype=torch.float32, device=self.device)

        _, model_groups = read_star_loop(fmodel, "model_groups")
        sum_groupnum = [int(d[Star.GroupNrParticles]) for d in model_groups]
        self.sum_groupnum = torch.Tensor(sum_groupnum).to(dtype=int, device=self.device)
        scales_group = [float(d[Star.GroupScaleCorrection]) for d in model_groups]
        self.scales_group = torch.Tensor(scales_group).to(dtype=torch.float32, device=self.device)
        group_names = [d[self.group_key] for d in model_groups]
        group_name_id = [int(d[Star.GroupNumber]) - 1 for d in model_groups]
        name2id = {group_names[i]:group_name_id[i] for i in range(len(group_names))} # new to new id

        sigma2_noise = []
        for i in group_name_id:
            block = "model_group_" + str(i+1)
            _, model_group = read_star_loop(fmodel, block)
            sigma2_noise.append([float(d[Star.Sigma2Noise]) for d in model_group])
        self.sigma2_noise = torch.Tensor(sigma2_noise).to(dtype=torch.float32, device=self.device)

        if self.rank==0: print("Processing fstar...")
        length = len(self.data_dicts)
        if Star.ClassNumber not in self.data_keys:
            self.class_ids_s1 = [1] * length
            self.data_keys.append(Star.ClassNumber)
        else:
            self.class_ids_s1 = [int(d[Star.ClassNumber]) for d in self.data_dicts]
        if Star.AnglePsi not in self.data_keys:
            self.angles_deg = [0.] * length
            self.data_keys.append(Star.AnglePsi)
        else:
            self.angles_deg = [float(d[Star.AnglePsi]) for d in self.data_dicts]
        if Star.OriginX not in self.data_keys:
            self.orignXs = [0.] * length
            self.data_keys.append(Star.OriginX)
        else:
            self.orignXs = [float(d[Star.OriginX]) for d in self.data_dicts]
        if Star.OriginY not in self.data_keys:
            self.orignYs = [0.] * length
            self.data_keys.append(Star.OriginY)
        else:
            self.orignYs = [float(d[Star.OriginY]) for d in self.data_dicts]
        if Star.NormCorrection not in self.data_keys:
            self.norms = [1.] * length
            self.data_keys.append(Star.NormCorrection)
        else:
            self.norms = [float(d[Star.NormCorrection]) for d in self.data_dicts]
        if Global.output_entropy:
            if Star.NrOfSignificantSamples not in self.data_keys:
                self.entropys = [0.] * length
                self.data_keys.append(Star.NrOfSignificantSamples)
            else:
                self.entropys = [float(d[Star.NrOfSignificantSamples]) for d in self.data_dicts]
                self.entropys = np.log2(self.entropys).tolist()

        # update group ids
        self.group_ids = [name2id[self.group_names[i]] for i in self.group_ids]
        self.group_names = group_names

        # like in initialize()
        self.mpi_initialize_continue() # only for mpi
        self.update_data_dicts()
        self.write(0)
        self.refs_init =self.refs.clone()
        self.tau2_class_T_init = self.tau2_class_T.clone()
        self.ssnr_init = self.ssnr.clone()

    def replace_refs(self, fref:str, ref_lp:float):
        """replace self.refs by fref, and lowpass to ref_lp"""
        if self.rank==0: print(f"Replace refs by {fref}, lowpass to {ref_lp} A")
        with mrcfile.open(fref, permissive=True) as mrc:
            refs = mrc.data.copy()
        if refs.ndim == 2:
            refs = refs[None, :, :]
        assert self.num_class == len(refs), f"num_class {self.num_class} != number of refs {len(refs)}"
        assert refs.shape[1] == self.ori_size and refs.shape[2] == self.ori_size, "ref size is different from particles"
        self.refs = torch.from_numpy(refs).to(dtype=torch.float32, device=self.device)
        # set resolution and lowpass and add mask
        current_size = int(self.ori_size * (2*self.pixel_size/ref_lp) + 0.5)
        self.updata_current_size(current_size)
        self.current_resolutions = torch.ones(self.num_class, device=self.device, dtype=torch.float32) * ref_lp
        self.refs = rfft2_center(self.refs)
        self.masker_rfft.set_soft_mask(self.rmax_cur, Global.edge_2, Mode.mask_zero)
        self.refs = self.masker_rfft.apply_mask(self.refs)
        self.refs = irfft2_center(self.refs)
        if not Global.skip_ref_mask:
            self.refs = self.masker_real.apply_mask(self.refs)
        #like in initialize_continue()
        spectrum_class = rfft2_center(self.refs)
        spectrum_class = spectrum_class.real**2 + spectrum_class.imag**2
        self.tau2_class_T = self.masker_rfft.radial_average(spectrum_class, self.rmax_ori) / 2
        self.tau2_class_T[:, 0] = self.tau2_class_T[:, 1]
        self.tau2_class_T *= self.T
        self.refs_init =self.refs.clone()
        self.tau2_class_T_init = self.tau2_class_T.clone()
        self.ssnr_init = self.ssnr.clone()

    def exp_wsum_dense(self, groupIds: Tensor1D, norms: Tensor1D, diffy: Tensor2D, diffx: Tensor2D, 
                       weight: Tensor4D, shift_img_r: Tensor4D, shift_img_i: Tensor4D, rot_ref2: Tensor4D, 
                       ctfs: Tensor3D, ctf2: Tensor3D, img2: Tensor3D, imgs2_ori: Tensor3D,
                       imgs_unmask: Tensor3D, ctfs_unmask: Tensor3D, noMW_ra: Tensor3D):
        """make expectation_batch() not too long\n
           contains sigma2_noise, norm correction, scale correction, sigma_offset\n
           will update the norms in place. will change rot_ref2"""
        # for sigma2_noise, and scale correction
        # weight*(ctf*rot_ref - shift_img)^2 = weight*ctf2*rot_ref2 + weight*shift_img2 - 2*ctf*(weight*rot_ref_r*shift_img_r + weight*rot_ref_i*shift_img_i)
        img_diff2: Tensor3D = oe.contract("ijkl,ijmn,klmn->imn", weight, shift_img_r, self.refs_rotated.real)
        img_diff2 += oe.contract("ijkl,ijmn,klmn->imn", weight, shift_img_i, self.refs_rotated.imag)
        img_diff2 *= (-2*ctfs)
        img_diff2 += oe.contract("ijkl,klmn->imn", weight, rot_ref2) * ctf2
        img_diff2 += oe.contract("ijkl,imn->imn", weight, img2)
        # sigma2_noise, fill the region out of the radius by spectrum
        mask: Tensor2D = self.masker_sigma.get_mask()>0
        _, _, ystart, yend, xstart, xend = get_yx_for_pad(self.current_size, self.ori_size, is_rfft=True)
        imgs2_ori[:, ystart:yend, xstart:xend][:, mask] = img_diff2[:, mask]
        if Global.ctf_mask_thres >= 0:
            imgs2_ori /= noMW_ra # for sigma2noise
            img_diff2 /= noMW_ra[:, ystart:yend, xstart:xend] # for norm correction
        self.wsum_diff2_group.index_add_(0, groupIds, imgs2_ori)

        # for norm correction
        norms *= torch.sqrt(2 * img_diff2.sum(dim=(1,2))) # *2 as Relion. because rfft has only half of the energy?
        norms.clamp_(1e-3, 1e3) # avoid too small or large value
        del img_diff2

        # for scale correction. each class has different ssnr, so need recalculate
        # crop to speedup
        # can skip it if has only one group
        if self.num_group > 1:
            ctfs = crop2D(ctfs, self.size_big_ssnr, is_rfft=True)
            ctf2 = crop2D(ctf2, self.size_big_ssnr, is_rfft=True)
            _, _, ystart, yend, xstart, xend = get_yx_for_pad(self.size_big_ssnr, self.current_size, is_rfft=True)
            rot_ref2 = rot_ref2[:,:,ystart:yend,xstart:xend]
            refs_rotated = self.refs_rotated[:,:,ystart:yend,xstart:xend].clone()
            shift_img_r = shift_img_r[:,:,ystart:yend,xstart:xend]
            shift_img_i = shift_img_i[:,:,ystart:yend,xstart:xend]
            mask = self.mask_small_ssnr[:,None,:,:].expand_as(refs_rotated) # size_big_ssnr
            refs_rotated[mask] = 0
            rot_ref2[mask] = 0  # will change rot_ref2 in place
            del mask
            sumXA = oe.contract("ijkl,ijmn,klmn->imn", weight, shift_img_r, refs_rotated.real)
            sumXA += oe.contract("ijkl,ijmn,klmn->imn", weight, shift_img_i, refs_rotated.imag)
            sumXA: Tensor1D = (sumXA*ctfs).sum(dim=(1,2)) / self.scales_group[groupIds]
            sumAA = oe.contract("ijkl,klmn->imn", weight, rot_ref2)
            sumAA: Tensor1D = (sumAA*ctf2).sum(dim=(1,2)) / self.scales_group[groupIds]**2
            self.wsum_XA_group.index_add_(0, groupIds, sumXA)
            self.wsum_AA_group.index_add_(0, groupIds, sumAA)
            del shift_img_r, shift_img_i

        # for sigma_offset
        weight_ij: Tensor2D = weight.sum(dim=(2,3))
        diff2: Tensor2D = diffy**2 + diffx**2
        self.wsum_offset_diff2 += float((weight_ij * diff2).sum())
        del weight_ij, diff2

        # for reconstruction. weight*invs2*ctf*shift_img and weight*invs2*ctf2. use unmasked img
        # not ignore negtive half y axis.
        # result will have hard mask, caused by invSigma2
        invs2 = self.invSigma2[groupIds] # no 1/2 here, and has origin point
        shift_img_r = oe.contract("imn,jmn->ijmn", imgs_unmask.real, self.offset_images.real)
        shift_img_r -= oe.contract("imn,jmn->ijmn", imgs_unmask.imag, self.offset_images.imag)
        shift_img_i = oe.contract("imn,jmn->ijmn", imgs_unmask.real, self.offset_images.imag)
        shift_img_i += oe.contract("imn,jmn->ijmn", imgs_unmask.imag, self.offset_images.real)
        use_entropy = Global.use_entropy >= 1 and self.niter >= Global.use_entropy
        # weight ijkl: batch, shift, class, ang
        if Global.output_probability:
            if Global.probability_max:
                class_prob = weight.max(dim=3)[0].max(dim=1)[0].cpu().tolist()
            else:
                class_prob = weight.sum(dim=(1,3)).cpu().tolist()
            self.class_prob += class_prob
        if use_entropy or Global.output_entropy:
            weight_entropy: Tensor1D = -weight.log2() * weight
            weight_entropy[torch.isnan(weight_entropy)] = 0.
            weight_entropy = weight_entropy.sum(dim=(1,2,3))
            if Global.output_entropy:
                self.entropys += weight_entropy.tolist()
            if use_entropy:
                entropy_max = np.log2(weight.size(1)*weight.size(2)*weight.size(3))
                weight_entropy = (entropy_max - weight_entropy) / entropy_max # from 0 to 1
                weight_entropy[weight_entropy<0.1] = 0.1 # avoid too small weight
                weight *= weight_entropy[:,None,None,None]
        self.wsum_data_unrot.real += oe.contract("ijkl,ijmn,imn->klmn", weight, shift_img_r, invs2*ctfs_unmask)
        self.wsum_data_unrot.imag += oe.contract("ijkl,ijmn,imn->klmn", weight, shift_img_i, invs2*ctfs_unmask)
        self.wsum_weight_unrot += oe.contract("ijkl,imn->klmn", weight, invs2*ctfs_unmask**2)

    def expectation_batch(self, imgs: Tensor3D, ctfs: Tensor3D, groupIds: Tensor1D, norms: Tensor1D, 
                          shiftys: Tensor1D, shiftxs: Tensor1D):
        # ijkl -> batch, shift, class, ang
        # ctf,invs2,img -> batch -> imn; offset_images -> n_shift -> jmn; shift_img -> ijmn; ref_rot -> class,ang -> klmn

        if Global.ctf_mask_thres >= 0:
            mask = ctfs.abs() <= Global.ctf_mask_thres
            noMW_ra = self.no_missing_wedge_average(mask)
            ctfs[mask] = 0.
        else:
            noMW_ra = None

        # scale correction and norm correction
        norms = torch.clamp(norms/self.norms_mean, 0.01, 100.0) # avoid too small or too large
        ctfs *= self.scales_group[groupIds][:,None,None]
        imgs /= norms[:,None,None]

        # to fourier space and crop to current_size
        imgs_unmask = rfft2_center(imgs) # for recon
        if Global.ctf_mask_thres >= 0: imgs_unmask[mask] = 0.
        imgs_unmask = crop2D(imgs_unmask, self.current_size, is_rfft=True)
        if not Global.skip_particle_mask:
            if Global.noise_particle_mask:
                backgrounds = imgs_phase_random(imgs)
                imgs = self.masker_real.apply_mask(imgs, bgs=backgrounds)
            else:
                imgs = self.masker_real.apply_mask(imgs) # bgmean
        imgs = rfft2_center(imgs)
        if Global.ctf_mask_thres >= 0: imgs[mask] = 0.
        imgs2_ori = imgs.real**2 + imgs.imag**2 # for sigma
        imgs = crop2D(imgs, self.current_size, is_rfft=True) # for alignment
        ctfs = crop2D(ctfs, self.current_size, is_rfft=True) # for alignment
        ctfs_unmask = ctfs.clone() # for recon
        # ignore negtive half y axis
        imgs[:, :imgs.size(1)//2, 0] = 0
        ctfs[:, :ctfs.size(1)//2, 0] = 0

        # shift on the fly. (img_real,img_imag)*(shift_real,shift_imag) = (img_real*shift_real-img_imag*shift_imag, img_real*shift_imag+img_imag*shift_real)
        shift_img_r: Tensor4D = oe.contract("imn,jmn->ijmn", imgs.real, self.offset_images.real)
        shift_img_r -= oe.contract("imn,jmn->ijmn", imgs.imag, self.offset_images.imag)
        shift_img_i: Tensor4D = oe.contract("imn,jmn->ijmn", imgs.real, self.offset_images.imag)
        shift_img_i += oe.contract("imn,jmn->ijmn", imgs.imag, self.offset_images.real)

        # some useful values
        invs2 = 1/2 * self.invSigma2[groupIds] # 1/2 because weight = 1/sqrt(2*pi*sigma2) * exp(-xxx2/2*sigm2)
        invs2 = self.masker_sigma.apply_mask(invs2) # ignore the region out of the radius
        invs2[:, invs2.size(1)//2, 0] = 0 # ignore origin point
        rot_ref2 = self.refs_rotated.real**2 + self.refs_rotated.imag**2
        img2 = imgs.real**2 + imgs.imag**2 # shift_img2 = img2
        ctf2 = ctfs**2

        # diff2 = (ctf*rot_ref - shift_img)^2 * invs2 = -2*invs2*ctf*rot_ref_r*shift_img_r - 2*invs2*ctf*rot_ref_i*shift_img_i + invs2*ctf2*rot_ref2 + invs2*shift_img2
        diff2: Tensor4D = oe.contract("imn,ijmn,klmn->ijkl", -2*invs2*ctfs, shift_img_r, self.refs_rotated.real) # ijkl: batch, shift, class, ang
        diff2 -= oe.contract("imn,ijmn,klmn->ijkl", 2*invs2*ctfs, shift_img_i, self.refs_rotated.imag)
        diff2 += oe.contract("imn,klmn->ikl", invs2*ctf2, rot_ref2)[:,None,:,:]
        diff2 += oe.contract("imn,imn->i", invs2, img2)[:,None,None,None]

        # get weight
        diff2_min = diff2.min(-1).values.min(-1).values.min(-1).values # min for each img
        diff2 -= diff2_min[:,None,None,None]
        weight: Tensor4D = torch.exp_(-diff2) # ijkl: batch, shift, class, ang
        diffy: Tensor2D = shiftys[:,None] + self.sample_y[None,:]
        diffx: Tensor2D = shiftxs[:,None] + self.sample_x[None,:]
        if Global.prior_classs:
            weight *= self.pdf_class[None,None,:,None]
        if Global.prior_offset and not Global.skip_align: # when skip_align, sigma_offset=0
            pdf_offset: Tensor2D = torch.exp(-(diffx**2+diffy**2)/(2*self.sigma_offset**2)) # ij
            pdf_offset /= pdf_offset.mean(dim=1, keepdim=True) # as Relion, but do not influence the result
            weight *= pdf_offset[:,:,None,None]
        weight /= weight.sum(dim=(1,2,3), keepdim=True) # normalize

        if Global.adaptive_fraction < 1: # we can skip it if not sparse, to save time
            thres = 1/(weight.size(1)*weight.size(2)*weight.size(3)) * (1-Global.adaptive_fraction) * 0.9
            weight[weight <= thres] = 0
            sig_num = (weight > 0).sum()
            if sig_num <= 0.01*weight.numel():
                remove_small_weight(weight, Global.adaptive_fraction)

        # count
        sum_classweight: Tensor1D = weight.sum(dim=(0,1,3)) # weight may be changed in exp_wsum_dense

        # many weighted sum. and change the norms. may change weight.
        self.exp_wsum_dense(groupIds, norms, diffy, diffx, weight, shift_img_r, shift_img_i, rot_ref2, 
                            ctfs, ctf2, img2, imgs2_ori, imgs_unmask, ctfs_unmask, noMW_ra)

        # count
        self.sum_weight += float(sum_classweight.sum())
        self.sum_groupnum.put_(groupIds, torch.ones_like(groupIds), accumulate=True)
        self.sum_classweight += sum_classweight
        del sum_classweight
        self.sum_norms += float(norms.sum())

        # update data
        id_shift, id_class, id_ang = argmax_weight(weight)
        del weight
        self.class_ids_s1 += (id_class + 1).tolist()
        self.angles_deg += torch.rad2deg(self.sample_ang[id_ang]).tolist()
        self.orignYs += (shiftys + self.sample_y[id_shift]).tolist()
        self.orignXs += (shiftxs + self.sample_x[id_shift]).tolist()
        self.norms += norms.tolist()

    def expectation(self, niter: int):
        # update current size
        res = self.current_resolutions.min()
        current_size = 2 * int(self.ori_size*self.pixel_size/res + Global.incr_size)
        if current_size > self.current_size:
            self.updata_current_size(current_size)

        # update sampling
        self.update_sampling(niter)

        # rotate refs
        # different from Relion, we do gridding correction after rotate and do not recalculate tau2_class_T
        self.refs_rotated = torch.zeros(self.num_class, len(self.sample_ang), self.current_size, self.current_size//2+1, 
                                        device=self.device, dtype=torch.complex64)
        refs_pad = self.masker_pad.apply_mask(self.refs)
        refs_pad = pad2D(refs_pad, self.ori_size_pad, is_rfft=False, fill=Mode.pad_corner)
        refs_pad = rfft2_center(refs_pad) # we irfft2 later, so not need padnorm
        rotater_ref = Rotation2D(self.sample_ang, self.ori_size_pad, is_rfft=True, direction=Mode.clockwise)
        for i,ref in enumerate(refs_pad):
            # rotate, unpad, mask, correction, crop
            # can just use rot_M_MN, but will cost more memory 
            ref = rotater_ref.rotate(ref, Mode.rot_1_N, complex=True)
            ref = irfft2_center(ref)
            ref = crop2D(ref, self.ori_size, is_rfft=False)
            ref = gridding_correction(ref, self.ori_size_pad, self.masker_real)
            if not Global.skip_ref_mask_E:
                ref = self.masker_real.apply_mask(ref) # already masked by masker_pad, so pad here seems unnecessary
            ref = rfft2_center(ref)
            self.refs_rotated[i] = crop2D(ref, self.current_size, is_rfft=True)
        del refs_pad, rotater_ref

        # prepare
        self.get_invSigma2()
        self.get_offset_images()
        self.wsum_diff2_group = torch.zeros(self.num_group, self.ori_size, self.ori_size//2+1, device=self.device, dtype=torch.float32)
        self.wsum_data_unrot = torch.zeros_like(self.refs_rotated)
        self.wsum_weight_unrot = torch.zeros_like(self.refs_rotated, dtype=torch.float32)
        if self.num_group > 1:
            self.wsum_XA_group = torch.zeros(self.num_group, device=self.device, dtype=torch.float32)
            self.wsum_AA_group = torch.zeros(self.num_group, device=self.device, dtype=torch.float32)
        self.wsum_offset_diff2 = 0.
        self.sum_weight = 0.
        self.sum_norms = 0.
        self.sum_groupnum = torch.zeros(self.num_group, device=self.device, dtype=int)
        self.sum_classweight = torch.zeros(self.num_class, device=self.device, dtype=torch.float32)

        dataset_exp = DatasetExpectation(self.data_dicts, self.group_ids, self.norms, self.orignYs, self.orignXs, wrap=Global.shift_wrap, pixel_size=self.pixel_size,
                                         img_dict=self.img_dict, ctf_dict=self.ctf_dict)
        dataloader_exp = DataLoader(dataset_exp, batch_size=Global.batch_exp, num_workers=Global.batch_worker, 
                                     pin_memory=Global.pin_memory, shuffle=False, drop_last=False)
        # will update class,x,y,ang,norm. maybe can save the old ones if want to compare. 
        self.class_ids_s1 = []
        self.orignYs = []
        self.orignXs = []
        self.angles_deg = []
        self.norms = []
        if Global.output_entropy:
            self.entropys = []
            if niter==1 and self.rank==0:
                print("Max possible NrOfSignificantSamples is:", len(self.sample_ang)*len(self.sample_x)*self.num_class) 
        if Global.output_probability:
            self.class_prob = []
        if self.rank==0: pbar = tqdm(total=len(dataloader_exp), desc=f"Expectation{self.niter:03d}")
        for imgs, ctfs, groupIds, norms, shiftys, shiftxs in dataloader_exp:
            imgs: Tensor3D = imgs.to(dtype=torch.float32, device=self.device)
            ctfs: Tensor3D = ctfs.to(dtype=torch.float32, device=self.device)
            groupIds: Tensor1D = groupIds.to(dtype=int, device=self.device)
            norms: Tensor1D = norms.to(dtype=torch.float32, device=self.device)
            shiftys: Tensor1D = shiftys.to(dtype=torch.float32, device=self.device)
            shiftxs: Tensor1D = shiftxs.to(dtype=torch.float32, device=self.device)
            if self.normalize:
                imgs -= imgs.mean(dim=(1,2), keepdim=True)
                imgs /= imgs.std(dim=(1,2), keepdim=True) + 1e-6
            self.expectation_batch(imgs, ctfs, groupIds, norms, shiftys, shiftxs)
            if self.rank==0: pbar.update(1)
        if self.rank==0: pbar.close()
        
        self.refs_rotated = None

        self.mpi_expectation() # only for mpi

        self.backrotate()
        self.wsum_data_unrot = None
        self.wsum_weight_unrot = None
        self.rotate_wsum_Cn()

    def backrotate(self):
        """backrotate wsum_data_unrot and wsum_weight_unrot to wsum_data and wsum_weight"""
        self.wsum_data = torch.zeros(self.num_class, self.current_size_pad, self.current_size_pad//2+1, device=self.device, dtype=torch.complex64)
        self.wsum_weight = torch.zeros(self.num_class, self.current_size_pad, self.current_size_pad//2+1, device=self.device, dtype=torch.float32)
        rotater = Rotation2D(self.sample_ang, self.current_size_pad, is_rfft=True, direction=Mode.anti_clockwise)
        for i, (data, weight) in enumerate(zip(self.wsum_data_unrot, self.wsum_weight_unrot)):
            # pad to current_size_pad
            if Global.mask_before_pad_f:
                data = self.masker_current_fft.apply_mask(data)
                weight = self.masker_current_fft.apply_mask(weight)
            data = irfft2_center(data)
            weight = irfft2_center(weight)
            if Global.mask_before_pad_r:
                data = self.masker_current_real.apply_mask(data)
                weight = self.masker_current_real.apply_mask(weight)
            data = pad2D(data, self.current_size_pad, is_rfft=False)
            data = rfft2_center(data) * self.current_padnorm # both * padnorm is unnecessary in fact
            weight = pad2D(weight, self.current_size_pad, is_rfft=False)
            weight = rfft2_center(weight).abs() * self.current_padnorm
            # rotate
            data = rotater.rotate(data, Mode.rot_N_N, complex=True).sum(dim=0)
            weight = rotater.rotate(weight, Mode.rot_N_N, complex=False).sum(dim=0)
            self.wsum_data[i] = data
            self.wsum_weight[i] = weight

    def rotate_wsum_Cn(self):
        if self.Cn <= 1:
            return
        rot_angles: Tensor1D = torch.linspace(0, 2*np.pi, self.Cn + 1, device=self.device, dtype=torch.float32)[:-1]
        rotater = Rotation2D(rot_angles, self.current_size_pad, is_rfft=True, direction=Mode.anti_clockwise)
        self.wsum_data = rotater.rotate(self.wsum_data, Mode.rot_M_MN, complex=True).sum(dim=1) / self.Cn
        self.wsum_weight = rotater.rotate(self.wsum_weight, Mode.rot_M_MN, complex=False).sum(dim=1) / self.Cn

    def maximization(self):
        if self.rank==0: print("Maximization...")
        # calculate sigma2_class here. wsum_weight will change when reconstruct
        self.sigma2_class.zero_()
        invweight: Tensor2D = self.masker_recon.radial_average(self.wsum_weight, self.rmax_cur, scale=self.current_pad)
        invweight = replace_nan_inf(1/invweight, 0.)
        if Global.mask_before_pad_f:
            # weight was masked, so correct it here
            mask1d = self.masker_current_fft.get_mask()[self.current_size//2:self.current_size//2+self.rmax_cur+2, 0]
            invweight *= mask1d[None, :]
        self.sigma2_class[:, :invweight.size(1)] = invweight
        self.data_vs_prior_class() # as Relion, use tau2 from last iteration

        # Wiener Filter like reconstruction and data over weight. add mask. update tau2_class_T
        self.reconstruction()
        if not Global.skip_ref_mask:
            self.refs = self.masker_real.apply_mask(self.refs)
        spectrum_class = rfft2_center(self.refs)
        spectrum_class = spectrum_class.real**2 + spectrum_class.imag**2
        self.tau2_class_T_old = self.tau2_class_T # saved for writing result
        self.tau2_class_T = self.masker_rfft.radial_average(spectrum_class, self.rmax_ori) / 2 # /2 because complex number
        self.tau2_class_T[:, 0] = self.tau2_class_T[:, 1] # avoid too small value for normalized images
        self.tau2_class_T *= self.T
        del spectrum_class

        # scale correction
        if self.num_group > 1:
            scales = self.wsum_XA_group / self.wsum_AA_group
            med = scales.median()
            scales.clamp_(med/Global.scale_range_thres, med*Global.scale_range_thres)
            mean = (scales*self.sum_groupnum).sum() / self.sum_groupnum.sum()
            self.scales_group = scales / mean

        # sigma2_noise
        self.sigma2_noise = self.masker_rfft.radial_average(self.wsum_diff2_group, self.rmax_ori) / 2 # /2 because complex number
        self.sigma2_noise[:, 0] = self.sigma2_noise[:, 1] # avoid too small value for normalized images
        self.sigma2_noise = fix_sigma2_noise(self.sigma2_noise) # in theory there will not have negtive value as in init
        self.sigma2_noise /= self.sum_groupnum[:,None]

        # other
        self.norms_mean = self.sum_norms / self.sum_weight
        self.sigma_offset = np.sqrt(self.wsum_offset_diff2 / self.sum_weight / 2) # /2 because 2D (x,y)

        # process class with no particle, back to init ref
        # print(self.sum_classweight)
        mask = self.sum_classweight < 1
        self.refs[mask,:,:] = self.refs_init[mask,:,:].clone()
        self.tau2_class_T[mask,:] = self.tau2_class_T_init[mask,:].clone()
        self.ssnr[mask,:] = self.ssnr_init[mask,:].clone()
        self.sigma2_class[mask,:] = 0
        self.sum_classweight[mask] = 1
        self.pdf_class = self.sum_classweight / self.sum_classweight.sum()

    def update_data_dicts(self):
        if Global.output_entropy:
            entropys = np.array(self.entropys)
            number = (2 ** entropys + 0.5).astype(int)
            for data, num in zip(self.data_dicts, number):
                data[Star.NrOfSignificantSamples] = f"{num:6d}"
        for data,c,a,x,y,n in zip(self.data_dicts, self.class_ids_s1, self.angles_deg, self.orignXs, self.orignYs, self.norms):
            data[Star.ClassNumber] = f"{c:5d}"
            data[Star.AnglePsi] = f"{a:8.2f}"
            data[Star.OriginX] = f"{x:8.4f}"
            data[Star.OriginY] = f"{y:8.4f}"
            data[Star.NormCorrection] = f"{n:8.4f}"

    def check_perturbation(self):
        if len(self.random_ang) < self.niter:
            perturb = random_perturb(self.niter-len(self.random_ang), init=self.random_ang[-1])
            self.random_ang = np.hstack((self.random_ang, perturb))
        if len(self.random_y) < self.niter:
            perturb = random_perturb(self.niter-len(self.random_y), init=self.random_y[-1])
            self.random_y = np.hstack((self.random_y, perturb))
        if len(self.random_x) < self.niter:
            perturb = random_perturb(self.niter-len(self.random_x), init=self.random_x[-1])
            self.random_x = np.hstack((self.random_x, perturb))
        if Global.skip_align or not Global.purturb:
            self.random_ang.fill(0)
            self.random_x.fill(0)
            self.random_y.fill(0)

    def iteration_one(self):
        self.niter += 1
        if self.niter > self.num_iter:
            if self.rank == 0: print("exceed the max iteration", self.niter, self.num_iter)
            self.check_perturbation()
        self.expectation(self.niter)
        self.maximization()
        self.update_data_dicts()
        self.write(self.niter)

    def iteration(self):
        for _ in range(self.num_iter):
            self.iteration_one()

    def write(self, niter:int):
        """write mrcs and star of initialization and initialize some value in data_dicts"""
        if self.rank == 0: print(f"writting...")
        fref = f"{self.out_pre}_it{niter:03d}_classes.mrcs"
        fstar_data = f"{self.out_pre}_it{niter:03d}_data.star"
        fstar_model = f"{self.out_pre}_it{niter:03d}_model.star"
        write_mrcs(self.refs.cpu().numpy(), fref, self.pixel_size)
        write_star_loop(self.data_keys, self.data_dicts, fstar_data, overwrite=True)

        if Global.output_probability and niter > 0:
            fprob = f"{self.out_pre}_it{niter:03d}_prob.txt"
            np.savetxt(fprob, self.class_prob, fmt="%10.3e")

        # model.star
        block = "model_general"
        dict_general = {
            Star.ReferenceDimensionality: str(2),
            Star.DataDimensionality: str(2),
            Star.OriginalImageSize: f"{self.ori_size:d}",
            Star.CurrentResolution: f"{self.current_resolutions.min():.2f}",
            Star.CurrentImageSize: f"{self.current_size:d}",
            Star.PaddingFactor: f"{self.pad:.2f}",
            Star.PixelSize: f"{self.pixel_size:.2f}",
            Star.NrClasses: f"{self.num_class:d}",
            Star.NrGroups: f"{self.num_group:d}",   
            Star.Tau2FudgeFactor: f"{self.T:.2f}",
            Star.NormCorrectionAverage: f"{self.norms_mean:.4f}",
            Star.SigmaOffsets: f"{self.sigma_offset:.4f}",
        }
        write_star_list(dict_general.keys(), dict_general, fstar_model, block, overwrite=True) # dict should save the order of keys

        block = "model_classes"
        dict0 = {
            Star.ReferenceImage: "",
            Star.ClassDistribution: "",
            Star.EstimatedResolution: ""
        }
        dict_list = []
        for i in range(self.num_class):
            dict_class = dict0.copy()
            dict_class[Star.ReferenceImage] = f"{i+1:06d}@{fref}"
            dict_class[Star.ClassDistribution] = f"{self.pdf_class[i]:10.6f}"
            dict_class[Star.EstimatedResolution] = f"{self.current_resolutions[i]:10.2f}"
            dict_list.append(dict_class)
        write_star_loop(dict0.keys(), dict_list, fstar_model, block, overwrite=False)

        block_pre = "model_class_"
        dict0 = {
            Star.SpectralIndex: "",
            Star.Resolution: "",
            Star.AngstromResolution: "",
            Star.SsnrMap: "",
            Star.ReferenceSigma2: "",
            Star.ReferenceTau2: ""
        }
        idx, res_f, res_r = self.get_resolutions()
        tau2_class = self.tau2_class_T_old / self.T # save the real tau
        for ic in range(self.num_class):
            block = block_pre + str(ic+1)
            dict_list = []
            for i in range(len(idx)):
                dict_class = dict0.copy()
                dict_class[Star.SpectralIndex] = f"{idx[i]:6d}"
                dict_class[Star.Resolution] = f"{res_f[i]:8.4f}"
                dict_class[Star.AngstromResolution] = f"{res_r[i]:8.2f}"
                dict_class[Star.SsnrMap] = f"{self.ssnr[ic,i]:10.3e}"
                dict_class[Star.ReferenceSigma2] = f"{self.sigma2_class[ic,i]:10.3e}"
                dict_class[Star.ReferenceTau2] = f"{tau2_class[ic,i]:10.3e}"
                dict_list.append(dict_class)
            write_star_loop(dict0.keys(), dict_list, fstar_model, block, overwrite=False)

        block = "model_groups"
        dict0 = {
            Star.GroupNumber: "",
            self.group_key: "",
            Star.GroupNrParticles: "",
            Star.GroupScaleCorrection: ""
        }
        dict_list = []
        for i in range(self.num_group):
            dict_group = dict0.copy()
            dict_group[Star.GroupNumber] = f"{i+1:6d}"
            dict_group[self.group_key] = self.group_names[i]
            dict_group[Star.GroupNrParticles] = f"{self.sum_groupnum[i]:6d}"
            dict_group[Star.GroupScaleCorrection] = f"{self.scales_group[i]:8.4f}"
            dict_list.append(dict_group)
        write_star_loop(dict0.keys(), dict_list, fstar_model, block, overwrite=False)

        block_pre = "model_group_"
        dict0 = {
            Star.SpectralIndex: "",
            Star.Resolution: "",
            Star.Sigma2Noise: ""
        }
        for ig in range(self.num_group):
            block = block_pre + str(ig+1)
            dict_list = []
            for i in range(len(idx)):
                dict_group = dict0.copy()
                dict_group[Star.SpectralIndex] = f"{idx[i]:6d}"
                dict_group[Star.Resolution] = f"{res_f[i]:8.4f}"
                dict_group[Star.Sigma2Noise] = f"{self.sigma2_noise[ig, i]:10.3e}"
                dict_list.append(dict_group)
            write_star_loop(dict0.keys(), dict_list, fstar_model, block, overwrite=False)

    def mpi_initialize(self):
        return
    
    def mpi_initialize_continue(self):
        return
    
    def mpi_expectation(self):
        return
    

class Class2D_mpi(Class2D):
    def __init__(self, data_dicts: List[Dict[str, str]], num_class: int, num_iter: int, out_pre: str, 
                 rank: int, worldsize: int, random_ang: Array1D, random_x: Array1D, random_y: Array1D, mpi_skip_init=False,
                 pixel_size: float=None, diameter: float=None,
                 psi_step=6., offset_step=1., offset_range=5., T=2., pad=2., 
                 init_max=0, normalize=False, Cn=1, img_ctf_dict:Tuple[dict]=None):
        
        super().__init__(data_dicts=data_dicts, 
                         num_class=num_class, 
                         num_iter=num_iter, 
                         out_pre=out_pre, 
                         device='cuda', # control by torch.cuda.set_device()
                         random_ang=random_ang, # all gpu use the same random
                         random_x=random_x, 
                         random_y=random_y,
                         pixel_size=pixel_size, 
                         diameter=diameter,
                         psi_step=psi_step, 
                         offset_step=offset_step, 
                         offset_range=offset_range, 
                         T=T, 
                         pad=pad,
                         init_max=init_max,
                         normalize=normalize,
                         Cn=Cn,
                         img_ctf_dict=img_ctf_dict,
                         rank=rank)
        
        self.worldsize = worldsize
        self.mpi_skip_init = mpi_skip_init
        # just split the data in order
        self.spliter: Array1D = np.linspace(0, len(self.data_dicts), self.worldsize+1, dtype=int)
        self.data_dicts_full = [data.copy() for data in self.data_dicts] # save it for write star
        self.group_ids_full = self.group_ids # used when mpi_skip_init
        if self.mpi_skip_init:
            if self.rank!=0:
                self.data_dicts = []
                self.group_ids = []
        else:
            self.data_dicts = self.data_dicts[self.spliter[self.rank]:self.spliter[self.rank+1]]
            self.group_ids = self.group_ids[self.spliter[self.rank]:self.spliter[self.rank+1]]
            if self.init_max > 0:
                self.init_max = self.init_max // worldsize + 1
        # things to gather on rank0 for writing star. list of list
        self.class_ids_s1_full: List[List[int]] = [None] * self.worldsize
        self.orignXs_full: List[List[float]] = [None] * self.worldsize
        self.orignYs_full: List[List[float]] = [None] * self.worldsize
        self.angles_deg_full: List[List[float]] = [None] * self.worldsize
        self.norms_full: List[List[float]] = [None] * self.worldsize
        if Global.output_entropy:
            self.entropys_full: List[List[float]] = [None] * self.worldsize
        if Global.output_probability:
            self.class_prob_full: List[List[List[float]]] = [None] * self.worldsize

    def reduce_float(self, number: float):
        number_list = [None] * self.worldsize 
        dist.all_gather_object(number_list, number)
        return float(np.array(number_list).sum())
    
    def reduce_complex(self, tensor: torch.Tensor):
        # my nccl and pytorch version do not support complex
        tensor_real = tensor.real.clone()
        tensor_imag = tensor.imag.clone()
        dist.all_reduce(tensor_real, op=dist.ReduceOp.SUM)
        dist.all_reduce(tensor_imag, op=dist.ReduceOp.SUM)
        return torch.complex(tensor_real, tensor_imag)

    def gather_data(self):
        dist.all_gather_object(self.class_ids_s1_full, self.class_ids_s1)
        dist.all_gather_object(self.orignXs_full, self.orignXs)
        dist.all_gather_object(self.orignYs_full, self.orignYs)
        dist.all_gather_object(self.angles_deg_full, self.angles_deg)
        dist.all_gather_object(self.norms_full, self.norms)
        if Global.output_entropy:
            dist.all_gather_object(self.entropys_full, self.entropys)
        if Global.output_probability and self.niter > 0:
            dist.all_gather_object(self.class_prob_full, self.class_prob)

    def sum_data_part(self):
        dist.all_reduce(self.sum_groupnum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.sum_classweight, op=dist.ReduceOp.SUM)
        self.sum_weight = self.reduce_float(self.sum_weight)
        self.sum_norms = self.reduce_float(self.sum_norms)

    def average_data(self):
        """each gpu should get the same result, so no need to merge. but just in case.\n
           will merge: refs, sigma2_noise, tau2_class_T, ssnr, scales_group."""
        if Global.mpi_average:
            dist.all_reduce(self.refs, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.sigma2_noise, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.tau2_class_T, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.ssnr, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.scales_group, op=dist.ReduceOp.SUM)
            self.refs /= self.worldsize
            self.sigma2_noise /= self.worldsize
            self.tau2_class_T /= self.worldsize
            self.ssnr /= self.worldsize
            self.scales_group /= self.worldsize
        else:
            dist.broadcast(self.refs, 0)
            dist.broadcast(self.sigma2_noise, 0)
            dist.broadcast(self.tau2_class_T, 0)
            dist.broadcast(self.ssnr, 0)
            dist.broadcast(self.scales_group, 0)

    def mpi_initialize(self):
        """will gather: class_ids_s1, orignXs, orignYs, angles_deg, norms.\n
           will sum: sum_init_fft, sum_init_fft2_group, wsum_data, wsum_weight, \n
           sum_weight, sum_groupnum, sum_classweight, sum_norms."""
        if Global.mpi_verbose: print(f"merge. rank {self.rank}")
        # gather
        self.gather_data()
        # sum
        self.sum_init_fft = self.reduce_complex(self.sum_init_fft)
        dist.all_reduce(self.sum_init_fft2_group, op=dist.ReduceOp.SUM)
        self.wsum_data = self.reduce_complex(self.wsum_data)
        dist.all_reduce(self.wsum_weight, op=dist.ReduceOp.SUM)
        self.sum_data_part()
        if self.mpi_skip_init:
            self.mpi_skip_init_recover() # distribute the work to each gpu
        return super().mpi_initialize() # nothing

    def mpi_initialize_continue(self):
        self.gather_data() # only data in fstar need to be gathered
        if self.mpi_skip_init:
            self.mpi_skip_init_recover()
        return super().mpi_initialize()
        
    def mpi_expectation(self):
        """will sum: wsum_diff2_group, wsum_XA_group, wsum_AA_group, wsum_offset_diff2,\n
           wsum_data, wsum_weight, sum_weight, sum_groupnum, sum_classweight, sum_norms.\n
           will gather: class_ids_s1, orignXs, orignYs, angles_deg, norms."""
        if Global.mpi_verbose: print(f"merge. rank {self.rank}")
        # gather
        self.gather_data()
        # sum
        dist.all_reduce(self.wsum_diff2_group, op=dist.ReduceOp.SUM)
        if self.num_group > 1:
            dist.all_reduce(self.wsum_XA_group, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.wsum_AA_group, op=dist.ReduceOp.SUM)
        self.wsum_data_unrot = self.reduce_complex(self.wsum_data_unrot)
        dist.all_reduce(self.wsum_weight_unrot, op=dist.ReduceOp.SUM)
        self.wsum_offset_diff2 = self.reduce_float(self.wsum_offset_diff2)
        self.sum_data_part() 
        return super().mpi_expectation() # nothing
    
    def mpi_skip_init_recover(self):
        # use at the end of mpi_initialize
        spliter, rank = self.spliter, self.rank
        self.data_dicts = self.data_dicts_full[spliter[rank]:spliter[rank+1]]
        self.group_ids = self.group_ids_full[spliter[rank]:spliter[rank+1]]
        
        self.class_ids_s1_full = [self.class_ids_s1_full[0][spliter[i]:spliter[i+1]] for i in range(self.worldsize)]
        self.orignXs_full = [self.orignXs_full[0][spliter[i]:spliter[i+1]] for i in range(self.worldsize)]
        self.orignYs_full = [self.orignYs_full[0][spliter[i]:spliter[i+1]] for i in range(self.worldsize)]
        self.angles_deg_full = [self.angles_deg_full[0][spliter[i]:spliter[i+1]] for i in range(self.worldsize)]
        self.norms_full = [self.norms_full[0][spliter[i]:spliter[i+1]] for i in range(self.worldsize)]
        if Global.output_entropy:
            self.entropys_full = [self.entropys_full[0][spliter[i]:spliter[i+1]] for i in range(self.worldsize)]
            self.entropys = self.entropys_full[rank]

        self.class_ids_s1 = self.class_ids_s1_full[rank]
        self.orignXs = self.orignXs_full[rank]
        self.orignYs = self.orignYs_full[rank]
        self.angles_deg = self.angles_deg_full[rank]
        self.norms = self.norms_full[rank]
        
    def write(self, niter: int):
        """update data_dicts_full and then write as usual"""
        self.average_data()
        if self.rank!=0:
            return
        
        for i in range(self.worldsize):
            # update each part of data_dicts_full
            self.data_dicts = self.data_dicts_full[self.spliter[i]:self.spliter[i+1]]
            self.class_ids_s1 = self.class_ids_s1_full[i]
            self.orignXs = self.orignXs_full[i]
            self.orignYs = self.orignYs_full[i]
            self.angles_deg = self.angles_deg_full[i]
            self.norms = self.norms_full[i]
            if Global.output_entropy:
                self.entropys = self.entropys_full[i]
            self.update_data_dicts()
        self.data_dicts = self.data_dicts_full
        if Global.output_probability and niter > 0:
            self.class_prob = []
            for cp in self.class_prob_full:
                self.class_prob += cp

        super().write(niter) # write only use data_dicts

        self.data_dicts_full = [data.copy() for data in self.data_dicts]
        # recover data for rank0
        self.data_dicts = self.data_dicts[self.spliter[self.rank]:self.spliter[self.rank+1]]
        self.class_ids_s1 = self.class_ids_s1_full[self.rank]
        self.orignXs = self.orignXs_full[self.rank]
        self.orignYs = self.orignYs_full[self.rank]
        self.angles_deg = self.angles_deg_full[self.rank]
        self.norms = self.norms_full[self.rank]
        if Global.output_entropy:
            self.entropys = self.entropys_full[self.rank]
        if Global.output_probability and niter > 0:
            self.class_prob = self.class_prob_full[self.rank]

    def check_perturbation(self):
        super().check_perturbation()
        if self.rank == 0:
            lst = [self.random_x, self.random_y, self.random_ang]
        else:
            lst = [None, None, None]
        dist.broadcast_object_list(lst, src=0)
        self.random_x, self.random_y, self.random_ang = lst


def get_mrcs_dict(data_dicts: List[Dict[str, str]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """get img_dict and ctf_dict (only for load_in_memory)"""
    print(f"Load mrcs into memory...")
    # img
    img_dict = {}
    for data in data_dicts:
        _, fname = get_mrc_idx_name(data[Star.ImageName])
        if fname not in img_dict:
            with mrcfile.open(fname, permissive=True) as mrc:
                data = mrc.data.astype(np.float32)
            img_dict[fname] = torch.from_numpy(data).share_memory_()
    # ctf
    if Star.CtfImage not in data_dicts[0]:
        return img_dict, None
    ctf_dict = {}
    for data in data_dicts:
        _, fname = get_mrc_idx_name(data[Star.CtfImage])
        if fname not in ctf_dict:
            with mrcfile.open(fname, permissive=True) as mrc:
                data = mrc.data.astype(np.float32)
            ctf_dict[fname] = torch.from_numpy(data).share_memory_()
    return img_dict, ctf_dict


def process_args(args: argparse.Namespace):
    # Global must be set in main function (after mp.spawn)
    Global.edge_real = args.soft_edge
    Global.noise_particle_mask = args.mask_noise
    Global.sigma_offset = args.offset_sigma
    Global.use_entropy = args.weight_entropy
    Global.ctf_mask_thres = args.ctf_thres
    Global.skip_align = args.skip_align
    Global.skip_ref_mask = args.skip_mask_ref
    Global.pin_memory = args.gpu!="cpu" and args.load_in_memory # pin_memory seems be useful only when load_in_memory
    Global.output_probability = args.out_prob in ["sum", "max"]
    Global.probability_max = args.out_prob == "max"


def main_single(data_dicts: dict, num_class: int, num_iter: int, out_pre: str, device: torch.device, pixel_size: float, diameter: float, 
                psi_step: float, offset_step: float, offset_range: float, T: float, pad: float, 
                init_max: int, normalize: bool, Cn: int, img_ctf_dict:Tuple[dict],
                fcontinue: str, fref: str, ref_lp:float, args: argparse.Namespace):
    process_args(args)
    print("particles number:", len(data_dicts))
    my_class2d = Class2D(data_dicts=data_dicts, 
                         num_class=num_class, 
                         num_iter=num_iter, 
                         out_pre=out_pre, 
                         device=device,
                         random_ang=None,
                         random_x=None,
                         random_y=None,
                         pixel_size=pixel_size, 
                         diameter=diameter, 
                         psi_step=psi_step, 
                         offset_step=offset_step, 
                         offset_range=offset_range, 
                         T=T, 
                         pad=pad,
                         init_max=init_max,
                         normalize=normalize,
                         Cn=Cn,
                         img_ctf_dict=img_ctf_dict)
    
    if fcontinue is None:
        my_class2d.initialize()
    else:
        my_class2d.initialize_continue(fcontinue)
    if fref is not None and ref_lp is not None:
        my_class2d.replace_refs(fref, ref_lp)
    my_class2d.iteration()


def main_multi(rank: int, worldsize: int, random_ang: Array1D, random_x: Array1D, random_y: Array1D, seed: int, addr: str, port: str, backend: str,
                data_dicts: dict, num_class: int, num_iter: int, out_pre: str, devices: List[int], pixel_size: float, diameter: float, 
                psi_step: float, offset_step: float, offset_range: float, T: float, pad: float, 
                init_max: int, normalize: bool, Cn: int, img_ctf_dict:Tuple[dict],
                fcontinue: str, fref: str, ref_lp:float, args: argparse.Namespace):
    process_args(args)
    # setup mpi
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(devices[rank])
    dist.init_process_group(backend=backend, rank=rank, world_size=worldsize)

    if seed is not None:
        set_seed(seed + rank) # different seed for each gpu
        skip_init = True
    else:
        skip_init = False

    if rank==0: print("particles number:", len(data_dicts))
    my_class2d = Class2D_mpi(data_dicts=data_dicts, 
                             num_class=num_class, 
                             num_iter=num_iter, 
                             out_pre=out_pre, 
                             rank=rank,
                             worldsize=worldsize,
                             random_ang=random_ang,
                             random_x=random_x,
                             random_y=random_y,
                             mpi_skip_init=skip_init,
                             pixel_size=pixel_size, 
                             diameter=diameter, 
                             psi_step=psi_step, 
                             offset_step=offset_step, 
                             offset_range=offset_range, 
                             T=T, 
                             pad=pad,
                             init_max=init_max,
                             normalize=normalize,
                             Cn=Cn,
                             img_ctf_dict=img_ctf_dict)
    
    if fcontinue is None:
        my_class2d.initialize()
    else:
        my_class2d.initialize_continue(fcontinue)
    if fref is not None and ref_lp is not None:
        my_class2d.replace_refs(fref, ref_lp)
    my_class2d.iteration()
    print("rank", rank, "finish.")
    dist.destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A standalone 2D classification program following Relion2's algorithm, with PyTorch GPU acceleration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    basic_group = parser.add_argument_group('Basic options')
    basic_group.add_argument('--fstar', '-i', type=str, required=True,
                        help='the input xx_data.star')
    basic_group.add_argument('--outpre', '-o', type=str, default="out/run",
                        help='prefix of all output files, will mkdir if not exist. should not end with "/"')
    basic_group.add_argument('--nclass', '-k', type=int, required=True,
                        help='number of class')
    basic_group.add_argument('--niter', '-n', type=int, default=25,
                        help='number of iteration')
    basic_group.add_argument('--gpu', '-g', type=str, default="cpu",
                        help='GPU Id. If use multi GPU, seperate them by ",", like "0,1,2"')
    basic_group.add_argument('--pixel', '-p', type=float, default=None,
                        help='you can specify pixel size in A, or will read from fstar')
    basic_group.add_argument('--diameter', '-d', type=float, default=None,
                        help='you can specify diameter in A, or will use box size')
    basic_group.add_argument('--mask_noise', action='store_true', 
                        help='useful for tomo projections. when masking particles, fill the outside region with phase randomized noise instead of mean.')
    basic_group.add_argument('--T', type=float, default=2.,
                        help='as Relion. but for tomo projections, set T=1 may be better')
    basic_group.add_argument('--normalize', action='store_true', 
                        help='Normalize input images so that the mean is 0 and std is 1, may cause problem if some images are nearly empty.')
    basic_group.add_argument('--Cn', type=int, default=1,
                        help='Cn symmetry for Class2D, default 1')
    basic_group.add_argument('--init_max', type=int, default=2000, 
                        help='maximum number of particles to use when initialization, <=0 means use all particles')
    basic_group.add_argument('--load_in_memory', action='store_true', 
                        help='Load all particles into memory to speed up.')
    basic_group.add_argument('--fcontinue', type=str, default=None,
                        help='can provide a _model.star to continue, and --fstar should be a _data.star')
    
    advanced_group = parser.add_argument_group('Advanced options')
    advanced_group.add_argument('--ctf_thres', type=float, default=1e-3,
                        help='<0 to skip. the region where abs(CTF)<=thres will be considered as missing wedge.')
    advanced_group.add_argument('--soft_edge', type=int, default=None,
                        help='soft edge width in pixel, when masking particles and refs. Default is 5*round(image_size/64)')
    basic_group.add_argument('--skip_mask_ref', action='store_true', 
                        help='Only masking particles, not reference.')
    advanced_group.add_argument('--psi_step', type=float, default=6.,
                        help='as Relion')
    advanced_group.add_argument('--offset_step', type=float, default=1.,
                        help='as Relion')
    advanced_group.add_argument('--offset_range', type=float, default=5.,
                        help='as Relion')
    advanced_group.add_argument('--offset_sigma', type=float, default=Global.sigma_offset,
                        help='inital sigma of offset prior, as Relion')
    advanced_group.add_argument('--seed', type=int, default=None,
                        help='set random seed if provided')
    advanced_group.add_argument('--weight_entropy', type=int, default=0, 
                        help='<1 to close. from this iteration, weight each particle by its information entropy during reconstruction')
    advanced_group.add_argument('--pad', type=float, default=2.0,
                        help='pad in real space before rotate')
    advanced_group.add_argument('--fref', type=str, default=None,
                        help='mrcs file. if provided, will use it as initial reference')
    advanced_group.add_argument('--ref_lp', type=float, default=None,
                        help='low pass filter for initial reference, in A. only work when --fref is provided')
    advanced_group.add_argument('--skip_align', action='store_true', 
                        help='set angles and shifts to zero, just classify')
    advanced_group.add_argument('--out_prob', type=str, choices=["sum", "max", "no"], default="no",
                        help='output class probability for each particle in _prob.txt, can be "sum" or "max"')
    
    mpi_group = parser.add_argument_group('MPI options (multi GPU, default values can work well in general)')
    mpi_group.add_argument('--backend', type=str, default="gloo",
                        help='nccl or gloo.')
    mpi_group.add_argument('--addr', type=str, default="localhost",
                        help='MASTER_ADDR')
    mpi_group.add_argument('--port', type=int, default=None,
                        help='MASTER_PORT')
    
    args = parser.parse_args()
    t0 = time()

    if (args.fref is not None and args.ref_lp is None) or (args.fref is None and args.ref_lp is not None):
        raise ValueError("--fref and --ref_lp should be provided together")

    # read star
    _, data_dicts = read_star_loop(args.fstar)
    if len(data_dicts) == 0:
        _, data_dicts = read_star_loop(args.fstar, "particles")

    if args.seed is not None:
        set_seed(args.seed)
    out_parent = Path(args.outpre).parent
    out_parent.mkdir(parents=True, exist_ok=True)

    fnote = out_parent/"note.txt"
    with open(fnote, "w") as f:
        json.dump(vars(args), f, indent=2)

    gpu_list = args.gpu.split(",")

    if args.skip_align:
        args.psi_step = 360. / args.Cn
        args.offset_step = 1.
        args.offset_range = 0.

    if args.load_in_memory:
        img_ctf_dict = get_mrcs_dict(data_dicts)
    else:
        img_ctf_dict = None

    if len(gpu_list) == 1:
        # single GPU (or CPU)
        device = gpu_list[0]
        if device != "cpu": device = int(device)
        main_single(data_dicts, args.nclass, args.niter, args.outpre, device, args.pixel, args.diameter,
                    args.psi_step, args.offset_step, args.offset_range, args.T, args.pad, 
                    args.init_max, args.normalize, args.Cn, img_ctf_dict,
                    args.fcontinue, args.fref, args.ref_lp, args)
        
    elif len(gpu_list) > 1:
        # single node multi GPU
        gpu_list = [int(gpu) for gpu in gpu_list]
        random_ang = random_perturb(args.niter)
        random_x = random_perturb(args.niter)
        random_y = random_perturb(args.niter)
        worldsize = len(gpu_list)
        if args.port is None:
            port = get_free_port(args.addr)
        else:
            port = args.port
        print("using multi GPU:", gpu_list, "port:", port)
        args_multi = (worldsize, random_ang, random_x, random_y, args.seed, args.addr, str(port), args.backend,
                      data_dicts, args.nclass, args.niter, args.outpre, gpu_list, args.pixel, args.diameter,
                      args.psi_step, args.offset_step, args.offset_range, args.T, args.pad, 
                      args.init_max, args.normalize, args.Cn, img_ctf_dict,
                      args.fcontinue, args.fref, args.ref_lp, args)
        mp.spawn(main_multi, args=args_multi, nprocs=worldsize) # the first arg will be rank

    else:
        print("the provided --gpu seems not right")

    print(f"time used: {(time()-t0)/60:.2f} min")
        