#!/usr/bin/env python3

# Copyright (C) 2024  Xiaofeng Yan, Shudong Li
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

import numpy as np
import mrcfile
from scipy.spatial.transform import Rotation as R
import argparse
import torch.cuda
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, NewType
import warnings
from Mpicker_2dprojection import ctfshift_warp, load_file, generate_thu, generate_star


Array1D = NewType("Array1D", np.ndarray)
Array2D = NewType("Array2D", np.ndarray)
Array3D = NewType("Array3D", np.ndarray)
Tensor1D = NewType("Tensor1D", torch.Tensor)
Tensor2D = NewType("Tensor2D", torch.Tensor)
Tensor3D = NewType("Tensor3D", torch.Tensor)
Tensor4D = NewType("Tensor4D", torch.Tensor)
Tensor5D = NewType("Tensor5D", torch.Tensor)


def prepare_data(coord_angle:Array2D, use_vector:bool, shiftz:Array1D) -> Tuple[Array2D, Array3D]:
    coords = coord_angle[:, 0:3]
    shiftz = shiftz
    angles = coord_angle[:, 3:6]
    if use_vector:
        n_vector = angles / np.linalg.norm(angles, axis=1, keepdims=True)
        tilt = np.arccos(n_vector[:, 2])
        psi = np.arctan2(n_vector[:, 1], -n_vector[:, 0])
        rot = np.zeros_like(tilt)
    else:
        rot, tilt, psi = np.deg2rad(angles).T
    angles = np.column_stack((-rot, -tilt, -psi))
    matrix = R.from_euler('zyz', angles).as_matrix()
    movexyz = matrix[:,:,2] * shiftz[:,None] # (batch, 3)*(batch, 1)
    coords += movexyz
    return coords, matrix


def prepare_ctf(ctf_path:str, ctf_x:float, ctf_y:float, ctf_z:float, shift_ctf:bool, flip_ctf:bool) -> Tuple[Array3D, List[float]]:
    with mrcfile.open(ctf_path, permissive=True) as mrc:
        tomo = mrc.data.astype(np.float32)
    if shift_ctf:
        tomo = ctfshift_warp(tomo)
    if flip_ctf:
        tomo = tomo[::-1]
        if ctf_x is not None: # user provide xyz
            ctf_z = tomo.shape[0] + 1 - ctf_z
    if ctf_x is None:
        ctf_x = tomo.shape[2] // 2 + 1
        ctf_y = tomo.shape[1] // 2 + 1
        ctf_z = tomo.shape[0] // 2 + 1
    return tomo, [ctf_x, ctf_y, ctf_z]


def interp_fill_batch(mgrid_batch:Tensor5D, tomos:Tensor4D) -> Tensor4D:
    mask = mgrid_batch.abs().max(dim=4).values <= 1
    if mask.all():
        return tomos
    nums = mask.sum(dim=(1,2,3))
    if (nums == 0).any():
        print("some particles are out of boundary, fill by zeros.")
        nums[nums == 0] = 1
    means = tomos.sum(dim=(1,2,3)) / nums
    means = means[:,None,None,None].expand_as(tomos)
    tomos[~mask] = means[~mask]
    return tomos


def interp_mgrid2tomo_batch(mgrid_batch:Tensor5D, tomo_batch:Tensor4D, pad:str) -> Tensor4D:
    mgrid_batch = mgrid_batch.permute(0,2,3,4,1) # (batch, zo, yo, xo, 3)
    mgrid_batch = mgrid_batch.flip(4) # to xyz
    tomo_batch.unsqueeze_(1) # (batch, 1, zi, yi, xi)
    # convert mgrid to [-1,1]
    mgrid_batch[:,:,:,:,0] *= 2/(tomo_batch.size(4)-1) # x
    mgrid_batch[:,:,:,:,1] *= 2/(tomo_batch.size(3)-1) # y
    mgrid_batch[:,:,:,:,2] *= 2/(tomo_batch.size(2)-1) # z
    mgrid_batch -= 1
    if pad == 'mean':
        result = F.grid_sample(tomo_batch, mgrid_batch, mode='bilinear', padding_mode='zeros', align_corners=True)
        result.squeeze_(1)
        result = interp_fill_batch(mgrid_batch, result) # little slower than just border
    else:
        result = F.grid_sample(tomo_batch, mgrid_batch, mode='bilinear', padding_mode=pad, align_corners=True)
        result.squeeze_(1)
    return result


def interp_mgrid_batch(coord:Tensor2D, matrix:Tensor3D, dxy:int, dz:int, 
                       batch:int, device:torch.device) -> Tensor5D:
    # coord start from 0
    # flip from xyz to zyx
    coord = coord.flip(1) # (batch, 3)
    matrix = matrix.flip(1,2) # (batch, 3, 3)
    # calculate z axis
    slicez = torch.arange(-(dz-1)/2, (dz+1)/2, device=device, dtype=torch.float32).expand(batch, -1)
    movez = matrix[:,:,0][:,:,None] * slicez[:,None,:] # (batch, 3, 1)*(batch, 1, z) -> (batch, 3, z)
    # calculate yx axis
    sliceyx = torch.arange(-(dxy//2), dxy-(dxy//2), device=device, dtype=torch.float32).expand(batch, -1)
    movey = matrix[:,:,1][:,:,None] * sliceyx[:,None,:] # (batch, 3, y)
    movex = matrix[:,:,2][:,:,None] * sliceyx[:,None,:] # (batch, 3, x)
    # add coord and zyx -> (batch, 3, z, y, x)
    mgridzyx = coord[:,:,None,None,None] + movez[:,:,:,None,None] + movey[:,:,None,:,None] + movex[:,:,None,None,:]
    return mgridzyx


def rotate_particle_batch(tomo:Tensor3D, coord:Tensor2D, matrix:Tensor3D, dxy:int, dz:int, pad:str) -> Tensor4D:
    # coord start from 0
    batch = len(coord)
    device = coord.device
    mgrid = interp_mgrid_batch(coord, matrix, dxy, dz, batch, device)
    tomo = tomo.expand(batch, -1, -1, -1)
    result = interp_mgrid2tomo_batch(mgrid, tomo, pad)
    return result


def splitter(coords_full:Array2D, coord_idx:Array1D, sizexyz:List[int], slimit:int, pad:int, 
             result:List[Tuple[Array1D,Array1D,Array1D]]) -> None:
    """provide [] as input, and result will be [[xyz_start,xyz_end,coord_idx],...]"""
    # coords start from 1
    if len(coord_idx) == 0:
        return
    coords = coords_full[coord_idx]
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    zyx_min_pad = np.maximum(xyz_min.astype(int) - pad - 1, 1)
    zyx_max_pad = np.minimum(xyz_max.astype(int) + pad + 2, sizexyz)
    if np.prod(zyx_max_pad - zyx_min_pad) <= slimit:
        result.append((zyx_min_pad - 1, zyx_max_pad, coord_idx))
    else:
        zyx = np.argmax(xyz_max - xyz_min)
        thres = (xyz_max[zyx] + xyz_min[zyx]) / 2
        mask = coords[:, zyx] < thres
        splitter(coords_full, coord_idx[mask], sizexyz, slimit, pad, result)
        splitter(coords_full, coord_idx[~mask], sizexyz, slimit, pad, result)


def work_func(tomo_full:Tensor3D, coords:Array2D, matrixs:Array2D, batch_size:int, dxy:int, dz:int,
               invert:bool, doparticle:bool, dotomo:bool, bar=True) -> Tuple[Array3D, Array3D]:
    gpuid = tomo_full.device
    result_stack = None
    result_sum = None

    if doparticle:
        result_stack = []
        dataset = TensorDataset(torch.Tensor(coords), torch.Tensor(matrixs))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if bar:
            dataloader = tqdm.tqdm(dataloader, desc="extract particle")
        for coord, matrix in dataloader:
            coord = coord.to(device=gpuid, dtype=torch.float32) - 1 # start from 1
            matrix = matrix.to(device=gpuid, dtype=torch.float32)
            tomos = rotate_particle_batch(tomo_full, coord, matrix, dxy, dz, 'mean')
            tomos = tomos.sum(axis=1)
            result_stack.append(tomos.cpu()) # might use too much memory
        result_stack = torch.cat(result_stack, dim=0)
        if invert:
            result_stack *= -1
        result_stack = result_stack.cpu().numpy().astype(np.float32)

    if dotomo:
        result_sum = torch.zeros(dxy, dxy, dxy, dtype=torch.float32, device=gpuid)
        dataset = TensorDataset(torch.Tensor(coords), torch.Tensor(matrixs))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if bar:
            dataloader = tqdm.tqdm(dataloader, desc="sum subtomo")
        for coord, matrix in dataloader:
            coord = coord.to(device=gpuid, dtype=torch.float32) - 1 # start from 1
            matrix = matrix.to(device=gpuid, dtype=torch.float32)
            tomos = rotate_particle_batch(tomo_full, coord, matrix, dxy, dxy, 'mean')
            tomos = tomos.sum(axis=0)
            result_sum += tomos
        if invert:
            result_sum *= -1
        result_sum = result_sum.cpu().numpy().astype(np.float32)

    return result_stack, result_sum


def main(args):
    data_path = args.data
    dxy = args.dxy
    dz = args.dz
    args_map = args.map
    args_ctf = args.ctf
    use_vector = args.use_vector
    batch_size = args.batch
    shiftz = args.shiftz
    flip_ctf = args.ctfflip
    shift_ctf = args.ctfshift
    ctfcenter = args.ctfcenter
    invert = args.invert
    output = args.output
    ctf_output = args.ctfout
    tomo_output = args.tomoout
    fthu = args.thu
    fstar = args.star
    conti = args.conti
    gpuid = args.gpuid
    if gpuid is None:
        gpuid = 'cpu'
    max_size = args.gb * 1024**3 / 4 # 4 bytes for float32

    coord_angle = np.loadtxt(data_path, ndmin=2, dtype=float)  # x y z nx ny nz (dx dy dz)
    if coord_angle.shape[1] >= 9:
        coord_angle[:, 0:3] -= coord_angle[:, 6:9] # realX = X - originX
    coord_angle = coord_angle[:, 0:6]
    shiftz = load_file(shiftz, len(coord_angle))
    shiftz = np.array([float(sz) for sz in shiftz])

    coords, matrixs = prepare_data(coord_angle, use_vector, shiftz)
    del coord_angle, shiftz
        
    if args_map is not None:
        doparticle = True
        map_path = args_map # just accept one tomo
        with mrcfile.mmap(map_path, permissive=True, mode='r') as mrc:
            size_tomo = mrc.data.shape
            voxel_size = mrc.voxel_size
    else:
        doparticle = False
        voxel_size = None

    if args_ctf is not None:
        doctf = True
        ctf_map_path = args_ctf # just accept one 3dctf
        if ctfcenter is None:
            ctf_x, ctf_y, ctf_z = None, None, None
        else:
            ctf_x, ctf_y, ctf_z = [float(i) for i in ctfcenter.split(',')]
        tomo_ctf, coords_ctf = prepare_ctf(ctf_map_path, ctf_x, ctf_y, ctf_z, shift_ctf, flip_ctf)
        tomo_ctf = torch.Tensor(tomo_ctf).to(device=gpuid, dtype=torch.float32)
        coords_ctf = np.array([coords_ctf] * len(coords))
    else:
        doctf = False

    if args_map is not None and tomo_output is not None:
        dotomo = True
    else:
        dotomo = False

    if doctf:
        result = []
        dataset = TensorDataset(torch.Tensor(coords_ctf), torch.Tensor(matrixs))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for coord, matrix in tqdm.tqdm(dataloader, desc="extract 2dctf"):
            coord = coord.to(device=gpuid, dtype=torch.float32) - 1 # start from 1
            matrix = matrix.to(device=gpuid, dtype=torch.float32)
            tomos = rotate_particle_batch(tomo_ctf, coord, matrix, dxy, 1, 'zeros')
            tomos = tomos[:, 0]
            result.append(tomos.cpu()) # might use too much memory
        result = torch.cat(result, dim=0)
        result = result.cpu().numpy().astype(np.float32)
        with mrcfile.new(ctf_output, overwrite=True) as mrc:
            mrc.set_data(result)
            if voxel_size is not None:
                mrc.voxel_size = voxel_size
            mrc.set_image_stack()

    if fthu is not None:
        if output is not None and ctf_output is not None:
            generate_thu(output, ctf_output, fthu, len(coords), conti)
        else:
            print("provide both output and ctfout to generate .thu file")
    if fstar is not None:
        if output is not None and ctf_output is not None:
            generate_star(output, ctf_output, fstar, len(coords), conti)
        else:
            print("provide both output and ctfout to generate .star file")

    if not doparticle and not dotomo:
        return
    
    size_pad = max(dxy, dz)
    if max_size < (2*size_pad) ** 3:
        max_size = (2*size_pad) ** 3
        print(f"--gb seems too small compare to dxy dz, change to {max_size*4/1024**3:.2f} GB.")
    split_result: List[Tuple[Array1D,Array1D,Array1D]] = []
    idx0 = np.arange(len(coords), dtype=int)
    size_xyz = size_tomo[::-1]
    splitter(coords, idx0, size_xyz, max_size, size_pad, split_result)

    result_stack = []
    result_sum = []
    result_idx = []
    for i in range(len(split_result)):
        xyz_start, xyz_end, coord_idx = split_result[i]
        print(f"Loading block {i+1}/{len(split_result)} ...")
        print(f"start_xyz: {xyz_start}, end_xyx: {xyz_end}, numbers: {len(coord_idx)}")
        # ignore warning caused by 'r' mode. seems faster than torch.from_numpy(xxx.copy()) ??
        warnings.filterwarnings("ignore")
        with mrcfile.mmap(map_path, permissive=True, mode='r') as mrc:
            xs, ys, zs = xyz_start
            xe, ye, ze = xyz_end
            tomo_full = torch.Tensor(mrc.data[zs:ze, ys:ye, xs:xe]).to(device=gpuid, dtype=torch.float32)
        warnings.resetwarnings()
        coord0 = coords[coord_idx] - xyz_start[None, :]
        matrix0 = matrixs[coord_idx]
        res_stack, res_sum = work_func(tomo_full, coord0, matrix0, batch_size, 
                                       dxy, dz, invert, doparticle, dotomo)
        result_stack.append(res_stack)
        result_sum.append(res_sum)
        result_idx.append(coord_idx)
        del tomo_full
        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    if doparticle:
        result_idx = np.concatenate(result_idx)
        idx_back = np.argsort(result_idx)
        result_stack = np.concatenate(result_stack, axis=0)[idx_back]
        with mrcfile.new(output, overwrite=True) as mrc:
            mrc.set_data(result_stack)
            if voxel_size is not None:
                mrc.voxel_size = voxel_size
            mrc.set_image_stack()    
    if dotomo:
        result_sum = np.sum(result_sum, axis=0)
        with mrcfile.new(tomo_output, overwrite=True) as mrc:
            mrc.set_data(result_sum / len(coords))
            if voxel_size is not None:
                mrc.voxel_size = voxel_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="faster 2dprojection using GPU, but only accept one input map (not txt).")
    parser.add_argument('--map', type=str,
                        help='one tomogram file (can not be .txt file)')
    parser.add_argument('--data', type=str, required=True,
                        help='file contains x,y,z,rot,tilt,psi, xyz start from 1 (in pixel), angels same as relion. or x,y,z,rot,tilt,psi,dx,dy,dz (rlnOriginXYZ)')
    parser.add_argument('--gpuid', type=int, default=None,
                        help='gpuid to use, default cpu')
    parser.add_argument('--gb', type=int, default=4,
                        help='max size (in GB) of tomo block that can be loaded in GPU at once. default 4GB')
    parser.add_argument('--dxy', type=int, default=50,
                        help='side length of output images, in pixel')
    parser.add_argument('--dz', type=int, default=1,
                        help='depth to project (thick), in pixel, default 1')
    parser.add_argument('--output', type=str, default='2d_output.mrcs',
                        help='output file, default 2d_output.mrcs')
    parser.add_argument('--shiftz', type=str, default="0",
                        help='shift the center along z when project(or a .txt file of shiftzs list), in pixel.')
    parser.add_argument('--invert', action='store_true',
                        help='invert the result projection, black to white')
    parser.add_argument('--ctf', type=str,
                        help='input one 3dctf(can not be .txt file) to generate 2dctfs. should have same size as dxy')
    parser.add_argument('--ctfout', type=str, default='2dctf_output.mrcs',
                        help='name of 2dctf file, default 2dctf_output.mrcs')
    parser.add_argument('--ctfcenter', type=str,
                        help='xyz center of 3dctf, start from 1, e.g "10,10,10", default use size//2+1')
    parser.add_argument('--ctfflip', action='store_true',
                        help='flipz for the 3dctf at first, useful when 3dctf generated by Relion2/3.')
    parser.add_argument('--ctfshift', action='store_true',
                        help='useful when 3dctf is generated by Warp or Relion4.')
    parser.add_argument('--thu', type=str,
                        help='generate .thu file for class2D in THUNDER2 if provided')
    parser.add_argument('--star', type=str,
                        help='generate .star file (rlnImageName and rlnCtfImage) for class2D if provided')
    parser.add_argument('--conti', action='store_true',
                        help='continue to write .thu or .star file if already exist. by default overwrite')
    parser.add_argument('--tomoout', type=str,
                        help='sum all subtomo to this mrc file if provided.')
    parser.add_argument('--use_vector', action='store_true',
                        help='provide norm vector nx ny nz in data, instead of rot tilt psi. same as rot=0')
    parser.add_argument('--batch', type=int, default=20,
                        help='batch size, default 20')
    args = parser.parse_args()
    main(args)