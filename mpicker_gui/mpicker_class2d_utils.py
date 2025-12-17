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

import numpy as np
import mrcfile
import torch
import torch.fft
import torch.cuda
import torch.nn.functional as F
from enum import Enum, auto
from typing import Dict, List, Tuple, Union, NewType
import socket


Array1D = NewType("Array1D", np.ndarray)
Array2D = NewType("Array2D", np.ndarray)
Array3D = NewType("Array3D", np.ndarray)
Tensor1D = NewType("Tensor1D", torch.Tensor)
Tensor2D = NewType("Tensor2D", torch.Tensor)
Tensor3D = NewType("Tensor3D", torch.Tensor)
Tensor4D = NewType("Tensor4D", torch.Tensor)
# complex
Tensor2D_c = NewType("Tensor2D_c", torch.Tensor)
Tensor3D_c = NewType("Tensor3D_c", torch.Tensor)
Tensor4D_c = NewType("Tensor4D_c", torch.Tensor)


class Mode(Enum):
    """Some modes"""
    # for Rotation2D
    rot_1_N = auto()
    rot_N_N = auto()
    rot_M_MN = auto()
    clockwise = auto() # rotate reference
    anti_clockwise = auto() # rotate particle
    # for mask background
    mask_zero = auto()
    mask_mean = auto() # mean of image
    mask_bgmean = auto() # mean of background part
    mask_fgmean = auto() # mean of part in mask
    pad_zero = auto()
    pad_corner = auto()


class Star:
    """Some global variables about starfile"""
    ImageName = "rlnImageName"
    CtfImage = "rlnCtfImage"
    GroupName = "rlnGroupName"
    OpticsGroup = "rlnOpticsGroup"
    MicrographName = "rlnMicrographName"
    Magnification = "rlnMagnification"
    DetectorPixelSize = "rlnDetectorPixelSize"
    ClassNumber = "rlnClassNumber"
    AnglePsi = "rlnAnglePsi"
    OriginX = "rlnOriginX"
    OriginY = "rlnOriginY"
    NormCorrection = "rlnNormCorrection"
    NrOfSignificantSamples = "rlnNrOfSignificantSamples"
    # for ctf if no CtfImage
    Voltage = "rlnVoltage"
    DefocusU = "rlnDefocusU"
    DefocusV = "rlnDefocusV"
    DefocusAngle = "rlnDefocusAngle"
    SphericalAberration = "rlnSphericalAberration"
    PhaseShift = "rlnPhaseShift"
    AmplitudeContrast = "rlnAmplitudeContrast"
    CtfBfactor = "rlnCtfBfactor"
    CtfScalefactor = "rlnCtfScalefactor"
    # for model.star
    ReferenceDimensionality = "rlnReferenceDimensionality"
    DataDimensionality = "rlnDataDimensionality"
    OriginalImageSize = "rlnOriginalImageSize"
    CurrentResolution = "rlnCurrentResolution"
    CurrentImageSize = "rlnCurrentImageSize"
    PaddingFactor = "rlnPaddingFactor"
    PixelSize = "rlnPixelSize"
    NrClasses = "rlnNrClasses"
    NrGroups = "rlnNrGroups"
    Tau2FudgeFactor = "rlnTau2FudgeFactor"
    NormCorrectionAverage = "rlnNormCorrectionAverage"
    SigmaOffsets = "rlnSigmaOffsets"
    ReferenceImage = "rlnReferenceImage"
    ClassDistribution = "rlnClassDistribution"
    EstimatedResolution = "rlnEstimatedResolution"
    SpectralIndex = "rlnSpectralIndex"
    Resolution = "rlnResolution"
    AngstromResolution = "rlnAngstromResolution"
    SsnrMap = "rlnSsnrMap"
    ReferenceSigma2 = "rlnReferenceSigma2"
    ReferenceTau2 = "rlnReferenceTau2"
    GroupNumber = "rlnGroupNumber"
    GroupNrParticles = "rlnGroupNrParticles"
    GroupScaleCorrection = "rlnGroupScaleCorrection"
    Sigma2Noise = "rlnSigma2Noise"


def read_star_loop(fname: str, block: str = "", once: bool = True) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a STAR file and return the data of specified block.

    Args:
        fname (str): The filename of the STAR file.
        block (str, optional): The block name in the STAR file. Defaults to "".
        once (bool, optional): If True, only return the data of the 1st appeared block. Defaults to True.

    Returns:
        keys (List[str]): The ordered keys of the loop data (of the last block if once=False).
        result (List[Dict[str, str]]): The data in the specified block.
    """
    flag_hit_block = False
    flag_hit_loop = False
    flag_hit_data = False
    result = []
    keys = []
    data_tmp = dict()
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue

            if line.startswith("data_"):
                if flag_hit_block: # already finished a block
                    if once:
                        return keys, result
                    else:
                        flag_hit_block = False
                        flag_hit_loop = False
                        flag_hit_data = False
                        keys = []
                        data_tmp = dict()
                        if line == "data_" + block:
                            flag_hit_block = True
                            print("Another specified block appears")
                        continue
                else:
                    if line == "data_" + block:
                        flag_hit_block = True
                    continue

            if flag_hit_block and not flag_hit_data:
                if line == "loop_":
                    flag_hit_loop = True
                    continue
                if flag_hit_loop:
                    if line.startswith("_"):
                        key = line.split()[0][1:]
                        keys.append(key)
                        data_tmp[key] = ""
                    else:
                        flag_hit_data = True

            if flag_hit_data:
                values = line.split()
                if len(values) != len(keys):
                    raise Exception(f"The number of columns does not match the number of keys:\n{line}")
                data = data_tmp.copy()
                for key, value in zip(keys, values):
                    data[key] = value
                result.append(data)
                continue
    return keys, result

def write_star_loop(keys: List[str], dataList: List[Dict[str, str]], fname: str, block="", overwrite = False):
    if overwrite:
        mode = "w"
    else:
        mode = "a"
    with open(fname, mode, newline="\n") as f: # force to use Linux style newline
        f.write("\n")
        if not overwrite:
            f.write("\n")
        f.write("data_" + block + "\n\n")
        f.write("loop_\n")
        for i,key in enumerate(keys):
            f.write("_" + key + " #" + str(i+1) + "\n")
        for data in dataList:
            values = [data[key] for key in keys]
            f.write("  ".join(values) + "\n")

def write_star_list(keys: List[str], data: Dict[str, str], fname: str, block="", overwrite = False):
    if overwrite:
        mode = "w"
    else:
        mode = "a"
    with open(fname, mode, newline="\n") as f: # force to use Linux style newline
        f.write("\n")
        if not overwrite:
            f.write("\n")
        f.write("data_" + block + "\n\n")
        for key in keys:
            f.write(f"{'_'+key:<35} {data[key]:>15}\n")

def read_star_list(fname: str, block: str = "") -> Tuple[List[str], Dict[str, str]]:
    """Read a STAR file and return the data of specified block as a dictionary.

    Args:
        fname (str): The filename of the STAR file.
        block (str, optional): The block name in the STAR file. Defaults to "".

    Returns:
        keys (List[str]): The ordered keys of the list data.
        result (Dict[str, str]): The data in the first specified block.
    """
    flag_hit_block = False
    keys = []
    result = dict()
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue

            if line.startswith("data_"):
                if flag_hit_block: # already finished a block
                    return keys, result
                else:
                    if line == "data_" + block:
                        flag_hit_block = True
                    continue

            if flag_hit_block:
                if line.startswith("_"):
                    parts = line.split()
                    key = parts[0][1:]
                    value = parts[1]
                    keys.append(key)
                    result[key] = value
                else:
                    continue
    return keys, result


def get_mrc_idx_name(fname: str) -> Tuple[int, str]:
    idx_fimg = fname.split('@')
    if len(idx_fimg) == 1:
        idx = None
        fimg = fname
    else:
        idx = int(idx_fimg[0]) - 1
        fimg = "@".join(idx_fimg[1:])
    return idx, fimg

def get_mrc_2d(fname: str) -> Array2D:
    idx, fimg = get_mrc_idx_name(fname)
    if idx is None:
        with mrcfile.open(fimg, permissive=True) as mrc:
            img = mrc.data.astype(np.float32).copy()
    else:
        with mrcfile.mmap(fimg, permissive=True) as mrc:
            img = np.array(mrc.data[idx], dtype=np.float32)
    return img

def write_mrcs(imgs: Array3D, fname: str, pixel_size: float, dtype=np.float32):
    with mrcfile.new(fname, overwrite=True) as mrc:
        mrc.set_data(imgs.astype(dtype))
        mrc.voxel_size = pixel_size
        mrc.set_image_stack()


def rfft2(imgs: torch.Tensor) -> torch.Tensor:
    # only norm when forward, as Relion
    return torch.fft.rfftn(imgs, dim=(-2,-1), norm="forward")

def irfft2(imgs: torch.Tensor) -> torch.Tensor:
    # not norm when backward, as Relion
    size = imgs.size(-2) # assume real image is sauqre
    return torch.fft.irfftn(imgs, dim=(-2,-1), norm="forward", s=(size, size))

def fftshift(x: Tensor1D) -> Tensor1D:
    # from fft center to image center
    return torch.roll(x, shifts=len(x)//2)

def fftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.roll(x, shifts=(x.size(-2)//2, x.size(-1)//2), dims=(-2,-1))

def rfftshift2(x: torch.Tensor) -> torch.Tensor:
    # I assume the origin point is (sy//2, 0) for all rfft in my code (start from 0) 
    return torch.roll(x, shifts=x.size(-2)//2, dims=-2)

def ifftshift(x: Tensor1D) -> Tensor1D:
    # from image center to fft center
    return torch.roll(x, shifts=-(len(x)//2))

def ifftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.roll(x, shifts=(-(x.size(-2)//2), -(x.size(-1)//2)), dims=(-2,-1))

def irfftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.roll(x, shifts=-(x.size(-2)//2), dims=-2)

def rfft2_center(imgs: Tensor3D) -> Tensor3D_c:
    """input and output both center on image center"""
    imgs = ifftshift2(imgs) # image center to fft center
    return rfftshift2(rfft2(imgs)) # fft center to image center

def irfft2_center(rffts: Tensor3D_c) -> Tensor3D:
    """input and output both center on image center"""
    rffts = irfftshift2(rffts) # image center to fft center
    return fftshift2(irfft2(rffts)) # fft center to image center

def kaiser_Fourier_value(x: torch.Tensor, a=1.9*2, alpha=15, norm=True) -> torch.Tensor:
    # here 339649.3699 is bessi0(alpha=15), just a constant
    # assuming -0.5<=x<=0.5, alpha>2*pi*a*x
    bessi0 = 339649.3699
    x = x.clone()
    x[torch.abs(x)>0.5] = 0.5
    sigma = torch.sqrt(alpha**2 - (2*np.pi*a*x)**2)
    bessi1_5 = torch.sqrt(2 / (np.pi*sigma))*(torch.cosh(sigma)-torch.sinh(sigma)/sigma)
    window = np.power(2*np.pi, 3/2) * np.power(a, 3) * bessi1_5 / (bessi0*torch.pow(sigma, 1.5))
    if norm:
        sigma = alpha
        bessi1_5 = np.sqrt(2 / (np.pi*sigma))*(np.cosh(sigma)-np.sinh(sigma)/sigma)
        window_max = np.power(2*np.pi, 3/2) * np.power(a, 3) * bessi1_5 / (bessi0*np.power(sigma, 1.5))
        window /= window_max
    return window

def sinc(x: torch.Tensor) -> torch.Tensor:
    y = torch.ones_like(x)
    mask = x!=0
    y[mask] = torch.sin(np.pi*x[mask]) / (np.pi*x[mask])
    return y


def replace_nan_inf(x: torch.Tensor, value=0.) -> torch.Tensor:
    """replace nan and inf by given value"""
    fill = torch.tensor(value, dtype=x.dtype, device=x.device)
    return torch.where(torch.isfinite(x), x, fill)


def get_yx_for_pad(si: int, so: int, is_rfft: bool) -> Tuple[int, int, int, int, int, int]:
    """output syo, sxo, ystart, yend, xstart, xend\n
       just used by pad2D and crop2D"""
    if is_rfft:
        syo, sxo = so, so//2+1
        sy, sx = si, si//2+1
        ystart = syo//2 - sy//2
        yend = ystart + sy
        xstart = 0
        xend = sx
    else:
        syo, sxo = so, so
        sy, sx = si, si
        ystart = syo//2 - sy//2
        yend = ystart + sy
        xstart = sxo//2 - sx//2
        xend = xstart + sx
    return syo, sxo, ystart, yend, xstart, xend

def pad2D(imgs: Tensor3D, sout:int, is_rfft:bool, fill=Mode.pad_zero) -> Tensor3D:
    syo, sxo, ystart, yend, xstart, xend = get_yx_for_pad(imgs.size(-2), sout, is_rfft)
    result = torch.zeros(len(imgs), syo, sxo, device=imgs.device, dtype=imgs.dtype)
    if fill == Mode.pad_corner:
        bgs = imgs[:, 0, 0]
        result += bgs[:, None, None]
    result[:, ystart:yend, xstart:xend] = imgs
    return result

def crop2D(imgs: Tensor3D, sout:int, is_rfft:bool) -> Tensor3D:
    _, _, ystart, yend, xstart, xend = get_yx_for_pad(sout, imgs.size(-2), is_rfft)
    return imgs[:, ystart:yend, xstart:xend]

def rfft_radius(size: int) -> int:
    """determine the max distance if fft image\n
       the length of radial average is this number + 1 (ssnr, sigma2, tau2...)"""
    return (size - 1) // 2


def fix_sigma2_noise(sigma2_noise: Tensor2D) -> Tensor2D:
    """replace the negtive value by the previous value as Relion"""
    no_neg = sigma2_noise >= 0
    values = sigma2_noise[no_neg]
    num_value = no_neg.sum(1, dtype=int) # each row has different number of no neg values
    if num_value.min() == 0:
        raise Exception("sigma2_noise has a row with all negtive values")
    idxs = torch.cumsum(no_neg, 1) - 1 # replace the negtive value by the nearest no neg value before it
    idxs[idxs < 0] = 0 # force the first value to be the first no neg value
    idxs[1:] += torch.cumsum(num_value[:-1], 0)[:,None]
    result = values[idxs]
    return result

def get_res_from_ssnr(ssnr: Tensor2D, resolutions: Tensor1D, thres=1.) -> Tensor2D:
    """find the position of the first ssnr<1 for each class, and return corresponding resolution one pixel before it\n
       will ignore the first point (zero frequency)\n
       if all ssnr<1, return the first resolution\n
       if all ssnr>=1, return the last resolution\n"""
    num_pixel = ssnr.size(1)
    low = ssnr<thres
    low[:, 0] = False # ignore zero frequency
    low = torch.cat([low, torch.ones_like(low[:,:1])], 1) # add a column of True at the end
    idx_low = torch.arange(0, num_pixel+1, device=ssnr.device, dtype=int).repeat(ssnr.size(0),1)
    idx_low = idx_low[low]
    num_low = low.sum(1, dtype=int)
    idx = torch.cumsum(num_low, 0).roll(1)
    idx[0] = 0
    idx = idx_low[idx] # save the position of the first low value for each class
    idx -= 1 # one pixel before
    idx[idx<0] = 0
    return resolutions[idx]

def get_res_from_ssnr_2(ssnr: Tensor2D, resolutions: Tensor1D, thres=1.) -> Tensor2D:
    """find the position of the last ssnr>=1 for each class and return corresponding resolutions\n
       if all ssnr<1, return the first resolution"""
    num_pixel = ssnr.size(1)
    high = ssnr>=thres
    high[:, 0] = True # at least one pixel for each class
    idx = torch.arange(0, num_pixel, device=ssnr.device, dtype=int).repeat(ssnr.size(0),1)
    idx = idx[high]
    num_high = high.sum(1, dtype=int)
    idx = idx[torch.cumsum(num_high, 0)-1] # save the position of the last high value for each class
    return resolutions[idx]


def random_perturb(length: int, perturbation_factor=0.5, init=0.) -> Array1D:
    perturbs = np.random.uniform(0.5*perturbation_factor, perturbation_factor, length)
    perturbs = np.cumsum(perturbs) + init
    perturbs = (perturbs + perturbation_factor) % 1 - perturbation_factor # wrap to +- perturb_factor
    return perturbs

def sample_angles(psi_step: float, perturb=0., dtype=torch.float32, Cn=1) -> Tensor1D:
    """step is in degree, output is in radian"""
    assert Cn >= 1
    Cn = int(Cn)
    num = int(360/Cn/psi_step + 0.5)
    psi = np.linspace(0., 2*np.pi/Cn, num+1)[:num]
    psi += np.deg2rad(perturb * psi_step)
    psi = (psi+np.pi) % (2*np.pi) - np.pi
    return torch.Tensor(psi).to(dtype)

def sample_offsets(offset_step: float, offset_range: float, perturbx=0., perturby=0., dtype=torch.float32) -> Tensor2D:
    """output 2 rows. x and y"""
    x = np.arange(-offset_range, offset_range+offset_step, offset_step).astype(float)
    y = np.arange(-offset_range, offset_range+offset_step, offset_step).astype(float)
    x, y = np.meshgrid(x, y)
    mask = x**2 + y**2 <= offset_range**2
    x = x[mask]
    y = y[mask]
    x += perturbx * offset_step
    y += perturby * offset_step
    xy = np.stack([x, y], axis=0)
    return torch.Tensor(xy).to(dtype)


def shift_img_int(img: Array2D, sy: int, sx: int, fill:float) -> Array2D:
    result = np.zeros_like(img) + fill
    sizey, sizex = img.shape
    if sy >= 0:
        ystarti = 0
        yendi = sizey - sy
        ystarto = sy
        yendo = sizey
    else:
        ystarti = -sy
        yendi = sizey
        ystarto = 0
        yendo = sizey + sy
    if sx >= 0:
        xstarti = 0
        xendi = sizex - sx
        xstarto = sx
        xendo = sizex
    else:
        xstarti = -sx
        xendi = sizex
        xstarto = 0
        xendo = sizex + sx
    result[ystarto:yendo, xstarto:xendo] = img[ystarti:yendi, xstarti:xendi]
    return result


def get_ctf_image(u2: Array2D, u4: Array2D, sin_2: Array2D, cos_2: Array2D, data: Dict[str, str]) -> Array2D:
    # same as Relion
    du = float(data[Star.DefocusU])
    dv = float(data.get(Star.DefocusV, du))
    ang = float(data.get(Star.DefocusAngle, 0))
    kv = float(data.get(Star.Voltage, 300)) * 1e3
    cs = float(data.get(Star.SphericalAberration, 0.01)) * 1e7
    ps = float(data.get(Star.PhaseShift, 0))
    q0 = float(data.get(Star.AmplitudeContrast, 0.1))
    bfac = float(data.get(Star.CtfBfactor, 0))
    scale = float(data.get(Star.CtfScalefactor, 1))

    d_ave = -(du+dv)/2
    d_dev = -(du-dv)/2
    ang = np.deg2rad(ang)
    lam = 12.2643247 / np.sqrt(kv * (1. + kv * 0.978466e-6))
    k1 = np.pi * lam
    k2 = np.pi/2 * cs * lam**3
    k4 = -bfac / 4.
    k5 = np.deg2rad(ps)

    # defocus = d_ave + d_dev*cos(2*ang), ang = ang_yx - ang
    # defocus = d_ave + d_dev*cos(2*ang)cos(2*ang_yx) + d_dev*sin(2*ang)sin(2*ang_yx)
    if du == dv:
        defocus = d_ave
    else:
        defocus = d_dev*np.cos(2*ang)*cos_2 + d_dev*np.sin(2*ang)*sin_2 + d_ave
    argument = k1 * defocus * u2 + k2 * u4 - (np.arcsin(q0) + k5)
    ctf = -scale*np.sin(argument)
    if k4!=0:
        ctf *= np.exp(k4*u2)
    return ctf


def remove_small_weight(weight: Tensor4D, ratio: float):
    """"weight has shape ijkl: batch, shift, class, angle\n
        for each batch the sum is 1, and the value>=0\n
        loop the batch and leave the biggest weights depend on the ratio"""
    # sort is slow
    for w in weight:
        values: Tensor1D = w[w > 0].sort(descending=True).values
        sumv = torch.cumsum(values, dim=0)
        idx = torch.where(sumv > ratio)[0]
        if len(idx) == 0:
            continue
        else:
            threshold = values[idx[0]]
            w[w < threshold] = 0

def argmax_weight(weight: Tensor4D) -> Tuple[Tensor1D, Tensor1D, Tensor1D]:
    """idx1,idx2,idx3 = weight.argmax(dims=(1, 2, 3))"""
    s0, s1, s2, s3 = weight.shape
    idx = weight.reshape(s0, -1).argmax(dim=1) # view fails when only 1 class?
    idx3 = idx % s3
    idx //= s3
    idx2 = idx % s2
    idx //= s2
    idx1 = idx
    return idx1, idx2, idx3


def back_radial_average(signal: Tensor2D, grid: Tensor2D, interp=False) -> Tensor3D:
    """such as use 1D invsigma2 to generate 2D invsigma2 image\n
       grid is the distance from the center, region out of signal filled by the last value\n
       interp means linear interpolation"""
    idx0: Tensor1D = torch.arange(len(signal), device=signal.device, dtype=int)
    idx1: Tensor2D = grid.clone()
    idx1[idx1 > signal.size(1)-1] = signal.size(1)-1
    if interp:
        idx1 = idx1.to(int)
        weight = grid - idx1
        value0 = signal
        value1 = torch.roll(signal, shifts=-1, dims=1)
        value1[:, -1] = signal[:, -1]
        result = value0[idx0[:,None,None], idx1[None,:,:]] * (1-weight[None,:,:]) 
        result += value1[idx0[:,None,None], idx1[None,:,:]] * weight[None,:,:]
    else:
        idx1 = (idx1 + 0.5).to(int)
        result = signal[idx0[:,None,None], idx1[None,:,:]]
    return result


def get_free_port(host):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def imgs_phase_random(imgs: Tensor3D) -> Tensor3D:
    """random the phase of each image in imgs independently\n
       imgs is real images, output is real images with same shape"""
    fimgs = rfft2(imgs).abs()
    noise = torch.polar(fimgs, torch.rand_like(fimgs)*(2*np.pi))
    noise[0, 0] = fimgs[0, 0] # origin point should be real
    noise = irfft2(noise)
    return noise


class Mask2D:
    def __init__(self, size: int, is_rfft: bool, device: torch.device, dtype=torch.float32):
        """if is_rfft, sy=size, sx=size//2+1, cy=size//2, cx=0.\n
           device and dtype is just for grid_dist and mask. make sure the images will be on the same device"""
        self.is_rfft: bool = is_rfft
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.sy: int
        self.sx: int
        self.cy: int
        self.cx: int
        self.set_size_center(size)
        self.grid_dist: Tensor2D
        self.set_grid_dist()
        # only after call set_soft_mask
        self.mask: Tensor2D 
        self.default_bgmode: Mode 

    def set_size_center(self, size: int):
        if self.is_rfft:
            self.sy, self.sx = size, size // 2 + 1
            self.cy, self.cx = size // 2, 0
        else:
            self.sy, self.sx = size, size
            self.cy, self.cx = size // 2, size // 2

    def set_grid_dist(self):
        y = torch.arange(-self.cy, self.sy - self.cy, device=self.device, dtype=self.dtype)
        x = torch.arange(-self.cx, self.sx - self.cx, device=self.device, dtype=self.dtype)
        grid_y, grid_x = torch.meshgrid(y, x)
        self.grid_dist = torch.sqrt(grid_y**2 + grid_x**2)

    def get_grid_dist(self) -> Tensor2D:
        """output the grid_dist itself, not clone"""
        return self.grid_dist
    
    def get_grid_yx(self) -> Tensor3D:
        """output stack(meshgrid(y, x))"""
        y = torch.arange(-self.cy, self.sy - self.cy, device=self.device, dtype=self.dtype)
        x = torch.arange(-self.cx, self.sx - self.cx, device=self.device, dtype=self.dtype)
        return torch.stack(torch.meshgrid(y, x))
    
    def get_mask(self) -> Tensor2D:
        """output the mask itself, not clone.\n
           mask only exits after call set_soft_mask()"""
        return self.mask

    def radial_average(self, imgs: Tensor3D, dmax: int, scale=1.) -> Tensor2D:
        """will round the distance, and then output the average for each distance start from 0 to dmax for
        each image, result shape will be (len(imgs), dmax+1)\n
        scale>1 means downsample. scale<1 means upsample (inaccurate)\n
        if is_rfft, ignore the negtive half y axis\n
        imgs dtype should be real"""
        index: Tensor2D = (self.grid_dist/scale+0.5).to(int) # same as Relion. can round 0.5 to 1
        # use the dmax+1 to store useless data for now
        index[index > dmax] = dmax + 1
        num = torch.bincount(index.flatten(), minlength=dmax+2)
        num[num == 0] = 1
        index: Tensor3D = index[None,:,:] + torch.arange(0, len(imgs)*(dmax+2), dmax+2, device=imgs.device)[:,None,None]
        sums = torch.bincount(index.flatten(), weights=imgs.flatten(), minlength=len(imgs)*(dmax+2)).reshape(len(imgs), dmax+2)
        results = sums[:, :dmax+1] / num[None, :dmax+1]
        return results

    def check_bg_mean_std(self, imgs: Tensor3D, radius: float) -> Tuple[Tensor1D, Tensor1D]:
        """return the mean and std of the part > radius for each image\n"""
        mask = (self.grid_dist > radius)
        bg = imgs[mask.expand(len(imgs),-1,-1)].reshape(len(imgs),-1)
        mean = bg.mean(axis=1)
        std = bg.std(axis=1)
        return mean, std

    def set_soft_mask(self, radius: int, edge=0, default_bgmode=Mode.mask_zero):
        """set the soft (if edge>0) cosine mask before you call self.apply_mask\n
           the mask will store as self.mask"""
        self.default_bgmode = default_bgmode
        result = torch.zeros_like(self.grid_dist)
        if edge > 0:
            mask = (self.grid_dist > radius) & (self.grid_dist < radius+edge)
            result[mask] = 1/2 + 1/2 * torch.cos(np.pi/edge * (self.grid_dist[mask]-radius))
        result[self.grid_dist <= radius] = 1
        self.mask = result

    def apply_mask(self, imgs: Tensor3D, bgmode: Mode=None, bgs: Tensor3D=None) -> Tensor3D:
        """bgmode means how to set the background value: mask_zero, mask_mean, or mask_bgmean\n
           need to generate mask by set_soft_mask at first\n"""
        if bgs is not None:
            mask_inv = 1 - self.mask
            result = imgs * self.mask[None,:,:] + bgs * mask_inv[None,:,:]
            return result
        if bgmode is None:
            bgmode = self.default_bgmode
        if bgmode == Mode.mask_zero:
            return imgs * self.mask[None,:,:]
        mask_inv = 1 - self.mask
        if bgmode == Mode.mask_mean:
            bg = imgs.mean(axis=(1,2))
        elif bgmode == Mode.mask_bgmean:
            bg = imgs * mask_inv[None,:,:]
            bg = bg.sum(axis=(1,2)) / mask_inv.sum()
        elif bgmode == Mode.mask_fgmean:
            bg = imgs * self.mask[None,:,:]
            bg = bg.sum(axis=(1,2)) / self.mask.sum()
        result = imgs * self.mask[None,:,:] + bg[:,None,None] * mask_inv[None,:,:]
        return result


def radial_average_slow(grid_dist: Tensor2D, imgs: Tensor3D, dmax: int, is_rfft=True) -> Tensor2D:
    # torch.bincount is better
    index: Tensor2D = (grid_dist+0.5).to(int)
    # use the (dmax+2)th column to store useless data for now
    num = torch.zeros(dmax + 2, device=imgs.device, dtype=imgs.dtype)
    sums = torch.zeros(len(imgs), dmax + 2, device=imgs.device, dtype=imgs.dtype)
    if is_rfft:
        index[:index.size(0)//2, 0] = dmax + 1
    index[index > dmax] = dmax + 1
    num.put_(index.flatten(), torch.ones_like(imgs[0]).flatten(), accumulate=True)
    num[num == 0] = 1
    index: Tensor3D = index[None,:,:] + torch.arange(0, sums.numel(), sums.size(1), device=imgs.device)[:,None,None]
    sums.put_(index.flatten(), imgs.flatten(), accumulate=True)
    results = sums[:, :dmax+1] / num[None, :dmax+1]
    return results


class Rotation2D:
    def __init__(self, angles: Tensor1D, size: int, size_out=None, scale=1., is_rfft=True, direction=Mode.anti_clockwise):
        """angles is in radian\n
           size_out equals size by default\n
           scale>1 means downsample, scale<1 means upsample\n
           if is_rfft, sx=size//2+1, True by default"""
        self.is_rfft: bool = is_rfft
        self.scale: float = scale
        # about size and center
        self.sy: int
        self.sx: int
        self.syo: int
        self.sxo: int
        self.cy: int
        self.cx: int
        self.cyo: int
        self.cxo: int
        self.cy_norm: float
        self.cx_norm: float
        self.set_size_center(size, size_out)
        # about rotation grids
        self.grids: Tensor4D # shape: N H W 2
        self.masks: Tensor3D # shape: N H W, only for rfft
        self.set_grids(angles, direction)

    def rotate(self, images: Union[Tensor2D, Tensor3D], mode: Mode, complex=True) ->  Union[Tensor3D, Tensor4D]:
        """rotate one image using many angles: rot_1_N\n
           or rotate each image using corresponding angle: rot_N_N\n
           or rotate many images using many angles: rot_M_MN\n"""
        if images.shape[-1] != self.sx or images.shape[-2] != self.sy:
            raise Exception(f"Image shape should be ({self.sy}, {self.sx}), not {images.shape[-2:]}")
        
        if mode == Mode.rot_1_N:
            if images.dim() != 2:
                raise Exception("rot_1_N mode requires 2D tensor")
            rotate_real = self.rotate_real_1_N
        elif mode == Mode.rot_N_N:
            if images.dim() != 3:
                raise Exception("rot_N_N mode requires 3D tensor")
            rotate_real = self.rotate_real_N_N
        elif mode == Mode.rot_M_MN:
            if images.dim() != 3:
                raise Exception("rot_M_MN mode requires 3D tensor")
            rotate_real = self.rotate_real_M_MN

        if self.is_rfft and mode == Mode.rot_M_MN:
            masks = self.masks.expand(len(images),-1,-1,-1)
        else:
            masks = self.masks

        if complex:
            imgs_real = rotate_real(images.real)
            imgs_imag = rotate_real(images.imag)
            if self.is_rfft:
                imgs_imag[masks] *= -1
            return torch.complex(imgs_real, imgs_imag)
        else:
            return rotate_real(images)
        
    def rotate_real_1_N(self, image: Tensor2D) -> Tensor3D:
        image = image.expand(len(self.grids),1,-1,-1)
        imgs = F.grid_sample(image, self.grids, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(1)
        return imgs
    
    def rotate_real_N_N(self, images: Tensor3D) -> Tensor3D:
        images = images[:,None,:,:]
        imgs = F.grid_sample(images, self.grids, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(1)
        return imgs
    
    def rotate_real_M_NM(self, images: Tensor3D) -> Tensor4D:
        images = images.expand(len(self.grids),-1,-1,-1)
        imgs = F.grid_sample(images, self.grids, mode='bilinear', padding_mode='zeros', align_corners=True)
        return imgs
    
    def rotate_real_M_MN(self, images: Tensor3D) -> Tensor4D:
        imgs = self.rotate_real_M_NM(images)
        imgs.transpose_(0,1) # from N C H W to imgs angs H W
        return imgs
    
    def set_size_center(self, size: int, size_out: int):
        """set the size and center of the image"""
        self.sy = size
        if self.is_rfft:
            self.sx = self.sy // 2 + 1
        else:
            self.sx = self.sy
        if size_out is None:
            self.syo = self.sy
            self.sxo = self.sx
        else:
            self.syo = size_out
            if self.is_rfft:
                self.sxo = self.syo // 2 + 1
            else:
                self.sxo = size_out
        # centering
        if self.is_rfft:
            self.cy, self.cx = self.sy//2, 0 # center of rfft
            self.cyo, self.cxo = self.syo//2, 0
        else:
            self.cy, self.cx = self.sy//2, self.sx//2
            self.cyo, self.cxo = self.syo//2, self.sxo//2
        # align_corners=True so use s-1 here
        self.cy_norm, self.cx_norm = 2*self.cy/(self.sy-1)-1, 2*self.cx/(self.sx-1)-1

    def set_grids(self, angles: Tensor1D, dirction: Mode):
        """generate the grids and masks for rotation. angles is in radian"""
        sy, sx, cy, cx = self.sy, self.sx, self.cy, self.cx
        syo, sxo, cyo, cxo = self.syo, self.sxo, self.cyo, self.cxo
        cy_norm, cx_norm = self.cy_norm, self.cx_norm
        pad = self.scale
        if dirction == Mode.anti_clockwise:
            # rotate image in anti-clockwise, so rotate grid in clockwise
            angles = -angles

        # matrix = [1/sx, 0; 0, 1/sy] * [cos, -sin; sin, cos] * [p, 0; 0, p] * [sxo, 0; 0, syo]
        r00 = (sxo-1)/(sx-1)*pad # pad > 1 means downsample, then sxo should < sx
        r01 = (syo-1)/(sx-1)*pad # align_corners=True so use s-1 here
        r10 = (sxo-1)/(sy-1)*pad
        r11 = (syo-1)/(sy-1)*pad
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        zeros = torch.zeros_like(angles)
        matrix = torch.stack([cos*r00, -sin*r01, zeros, sin*r10, cos*r11, zeros], dim=-1).reshape(-1, 2, 3)
        grids = F.affine_grid(matrix, (len(angles),1,syo,sxo), align_corners=True)

        # centering
        center = grids[:,cyo,cxo,:][:,None,None,:].clone()
        grids[:,:,:,0] -= center[:,:,:,0] - cx_norm
        grids[:,:,:,1] -= center[:,:,:,1] - cy_norm

        if self.is_rfft:
            # apply symmetry, x,y -> -x,-y; for x < 0 or (x == 0 and y > size//2)
            # assume image(0,y>size//2) shoud be conjugate to image(0,y<size//2), 
            # should check Hermitian symmetry of y axis after rotations
            masks = (grids[:,:,:,0] < cx_norm)
            grids[masks,:] *= -1
            grids[:,:,:,0][masks] += 2 * cx_norm
            grids[:,:,:,1][masks] += 2 * cy_norm
        else:
            masks = None

        self.grids = grids
        self.masks = masks