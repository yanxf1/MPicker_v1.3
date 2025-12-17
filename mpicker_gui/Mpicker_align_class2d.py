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
import argparse
from typing import Dict, List, Tuple, Union, NewType
from scipy.ndimage import map_coordinates
from tqdm import tqdm

Array1D = NewType("Array1D", np.ndarray)
Array2D = NewType("Array2D", np.ndarray)
Array3D = NewType("Array3D", np.ndarray)
Array4D = NewType("Array4D", np.ndarray)


def rotate_img(img:Array2D, angles:Array1D, coords:Array2D, fill:float=None) -> Array4D:
    """
    Rotate the image around the given coords with given angles and then shifts.
    anti-clockwise is positive.
    Args:
        img: 2D array, read by mrcfile, square
        angles: 1D array, in degrees
        coords: 2D array, N*4, cx cy sx sy. cxy are rotation center, start from 0, sxy are shift.
        fill: float, fill value. if None, fill outside by the mean of inside
    Returns:
        4D array. img_rotate = result[id_ang,id_coord,:,:]
    """
    if fill is None:
        fill = 0
        fill_mean = True
    else:
        fill_mean = False
    length = img.shape[0]
    cos = np.cos(np.deg2rad(angles))
    sin = np.sin(np.deg2rad(angles))
    cx, cy, sx, sy = coords.T
    axisy = np.arange(length, dtype=float) - (cy+sy)[:,None] # (num_coord, leny)
    axisx = np.arange(length, dtype=float) - (cx+sx)[:,None] # (num_coord, lenx)
    movey = np.stack((cos, sin))[:,:,None,None] * axisy[None,None,:,:] # (2, num_ang, num_coord, leny)
    movex = np.stack((-sin, cos))[:,:,None,None] * axisx[None,None,:,:] # (2, num_ang, num_coord, lenx)
    backyx = np.stack((cy, cx)) # (2, num_coord)
    mgridyx = backyx[:,None,:,None,None] + movey[:,:,:,:,None] + movex[:,:,:,None,:] # (2, num_ang, num_coord, leny, lenx)
    del axisy, axisx, movey, movex, backyx
    results = map_coordinates(img, mgridyx, order=1, mode='constant', cval=fill, 
                              prefilter=False, output=img.dtype) # (num_ang, num_coord, leny, lenx)
    if fill_mean:
        mask = (mgridyx[0] >= 0) & (mgridyx[0] < length) & (mgridyx[1] >= 0) & (mgridyx[1] < length)
        del mgridyx
        nums = np.maximum(mask.sum(axis=(2,3)), 1)
        means = results.sum(axis=(2,3)) / nums
        means = np.broadcast_to(means[:,:,None,None], results.shape)
        results[~mask] = means[~mask]
    return results


def soft_edge(dist:np.ndarray, sigma:float) -> np.ndarray:
    dist = dist.copy()
    dist[dist < 0] = 0
    if sigma > 0:
        mask = np.exp(-dist**2/(2*sigma**2))
    else:
        mask = dist
        mask[mask > 0] = 1
    return mask


def lowpass(imgs:Array3D, cutoff:float, sigma:float) -> Array3D:
    """
    Low pass filter the images
    Args:
        imgs: 3D array, square
        cutoff: float, cutoff frequency, 0 to 0.5
        sigma: float, gaussian sigma in pixels
    Returns:
        3D array, filtered images
    """
    length = imgs.shape[1]
    freqsx = np.fft.rfftfreq(length)
    freqsy = np.fft.fftfreq(length)
    freqs = np.sqrt(freqsy[:,None]**2 + freqsx[None,:]**2)
    dist = length * (freqs - cutoff)
    mask = soft_edge(dist, sigma)
    results = np.fft.rfftn(imgs, axes=(1,2)) * mask[None,:,:]
    results = np.fft.irfftn(results, axes=(1,2))
    return results


def add_mask(imgs:Array3D, radius:float, cx:float, cy:float, sigma:float) -> Array3D:
    """
    Add a mask to the images
    Args:
        imgs: 3D array, square
        radius: float, radius of the mask in pixels
        cx_cy: float, center coordinate, start from 0
        sigma: float, gaussian sigma in pixels
    Returns:
        3D array, masked images
    """
    length = imgs.shape[1]
    axisx = np.arange(length, dtype=float) - cx
    axisy = np.arange(length, dtype=float) - cy
    dist = np.sqrt(axisy[:,None]**2 + axisx[None,:]**2) - radius
    mask = soft_edge(dist, sigma)
    results = imgs * mask[None,:,:]
    return results

    
def compare_scores(refs:Array3D, imgs:Array3D) -> Array2D:
    """
    Compare the scores of the images with the references
    Args:
        refs: 3D array, square
        imgs: 3D array, square
    Returns:
        2D array, score = result[id_ref, id_img]
    """
    normr = np.sqrt((refs**2).sum(axis=(1,2)))
    normi = np.sqrt((imgs**2).sum(axis=(1,2)))
    scores = np.einsum('imn,jmn->ij', refs, imgs)
    scores /= normr[:,None] * normi[None,:]
    return scores


def mass_centerxy(img:Array2D) -> Tuple[float, float]:
    img = img.copy()
    img -= np.median(img)
    img[img < 0] = 0
    length = img.shape[0]
    axisx = np.arange(length, dtype=float)
    axisy = np.arange(length, dtype=float)
    cx = (img * axisx[None,:]).sum() / img.sum()
    cy = (img * axisy[:,None]).sum() / img.sum()
    return cx, cy


def main(args):
    fin = args.i
    fout = args.o
    fmout = args.mo
    ids = args.ids
    fref = args.ref
    refxy = args.refxy
    from1 = args.one
    lp = args.lp
    diameter = args.m
    ang = args.ang
    shift = args.shift
    srange = args.range
    shift_cx, shift_cy = map(float, args.scenter.split(','))
    ang2 = args.ang2
    shift2 = args.shift2
    addang = args.addang
    sigma = args.sigma

    with mrcfile.open(fin, mode='r', permissive=True) as mrc:
        imgs_ori = mrc.data.copy()
        voxel = mrc.voxel_size
    if ids is not None:
        ids = np.array([int(i) for i in ids.split(',')])
        if from1:
            ids -= 1
    else:
        ids = np.arange(len(imgs_ori))
    imgs = imgs_ori[ids].copy()

    if fref is None:
        ref = imgs[0].copy()
    else:
        try:
            fref = int(fref)
            if from1:
                fref = int(fref) - 1
            ref = imgs_ori[fref].copy()
        except:
            with mrcfile.open(fref, mode='r', permissive=True) as mrc:
                ref = mrc.data.copy()

    if refxy is not None:
        cx, cy = map(float, refxy.split(','))
    else:
        cx, cy = mass_centerxy(ref)

    if diameter is not None:
        radius = diameter / 2
    else:
        radius = min(cx, cy, len(ref)-cx, len(ref)-cy)
    print(f"cx, cy, and mask_radius: {cx:.2f}, {cy:.2f}, {radius:.2f}")

    # low pass filter
    imgs = lowpass(imgs, lp, sigma)
    ref = lowpass(ref[None,:,:], lp, sigma)[0]

    # rot ref
    center = len(ref) // 2
    angles_global = np.linspace(0, 360, int(360/ang+0.5), endpoint=False)
    coords_ref = np.array([[cx, cy, center-cx, center-cy]])
    ref_rots = rotate_img(ref, angles_global, coords_ref, fill=0)[:,0,:,:]
    ref_rots = add_mask(ref_rots, radius, center, center, sigma)
    angles_local0 = np.arange(-ang+ang2, ang, ang2)

    # shift
    sx, sy = np.mgrid[-srange:srange+shift:shift, -srange:srange+shift:shift]
    mask = np.sqrt(sx**2 + sy**2) <= srange
    sx = sx[mask] + shift_cx
    sy = sy[mask] + shift_cy
    coords_global = np.column_stack((np.zeros_like(sx), np.zeros_like(sx), sx, sy))
    sx, sy = np.mgrid[-shift+shift2:shift:shift2, -shift+shift2:shift:shift2]
    sx = sx.flatten()
    sy = sy.flatten()
    coords_local0 = np.column_stack((np.zeros_like(sx), np.zeros_like(sx), sx, sy))

    results = []
    for idx_this, img in tqdm(zip(ids, imgs), total=len(ids)):
        # global search
        img_shifts = rotate_img(img, np.array([0]), coords_global, fill=0)[0,:,:,:]
        img_shifts = add_mask(img_shifts, radius, center, center, sigma)
        scores = compare_scores(ref_rots, img_shifts)
        id_ang, id_shift = np.unravel_index(np.argmax(scores), scores.shape)
        ang_best = angles_global[id_ang]
        sx_best, sy_best = coords_global[id_shift, 2:4]
        # local search
        if ang2 < ang and shift2 < shift:
            angles_local = angles_local0 + ang_best
            ref_rots_local = rotate_img(ref, angles_local, coords_ref, fill=0)[:,0,:,:]
            ref_rots_local = add_mask(ref_rots_local, radius, center, center, sigma)
            coords_local = coords_local0.copy()
            coords_local[:,2] += sx_best
            coords_local[:,3] += sy_best
            img_shifts_local = rotate_img(img, np.array([0]), coords_local, fill=0)[0,:,:,:]
            img_shifts_local = add_mask(img_shifts_local, radius, center, center, sigma)
            scores = compare_scores(ref_rots_local, img_shifts_local)
            id_ang, id_shift = np.unravel_index(np.argmax(scores), scores.shape)
            ang_best = angles_local[id_ang]
            sx_best, sy_best = coords_local[id_shift, 2:4]
        if from1:
            idx_this += 1
        ang_best = -ang_best # we need the angle of img
        ang_best = (ang_best + addang) % 360
        if ang_best > 180:
            ang_best -= 360
        results.append([idx_this, sx_best, sy_best, ang_best])

    print("writing results")
    results = np.array(results)
    np.savetxt(fout, results, fmt='%3d %6.2f %6.2f %7.2f')

    if fmout is not None:
        rot_results = []
        for img, result in zip(imgs_ori[ids], results):
            _, sx, sy, ang = result
            img_rot = rotate_img(img, np.array([ang]), np.array([[center-sx, center-sy, sx, sy]]), fill=0)[0,0,:,:]
            rot_results.append(img_rot)
        rot_results = np.array(rot_results, dtype=np.float32)
        with mrcfile.new(fmout, overwrite=True) as mrc:
            mrc.set_data(rot_results)
            mrc.voxel_size = voxel
            mrc.set_image_stack()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="align the result of Class2D", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--i', type=str, required=True, 
                        help='input mrcs file')
    parser.add_argument('--o', type=str, required=True, 
                        help='output align result. 4 columns, cls_id dx dy rot. just the fmove in Mpicker_convert_class2d.py')
    parser.add_argument('--mo', type=str, default=None,
                        help='output aligned image in mrcs.')
    parser.add_argument('--ids', type=str, default=None,
                        help='class ids to align, such as "2,3,0", defalut align all')
    parser.add_argument('--ref', type=str, default=None,
                        help='class id of the reference, or a mrc file, default use the first')
    parser.add_argument('--refxy', type=str, default=None,
                        help='the xy center of the reference, start from 0, such as "64,64". default use mass center.')
    parser.add_argument('--one', action='store_true',
                        help='if you want class id (in --o --ids --ref) start from 1, by default start from 0')
    parser.add_argument('--lp', type=float, default=0.3,
                        help='low pass filter cutoff, Nyquist frequency is 0.5')
    parser.add_argument('--m', type=float, default=None,
                        help='the diameter of the mask, in pixel')
    parser.add_argument('--ang', type=float, default=2,
                        help='rotation angle step, in degree')
    parser.add_argument('--shift', type=float, default=1,
                        help='shift step, in pixel')
    parser.add_argument('--range', type=float, default=10,
                        help='max radius to search the shift, in pixel')
    parser.add_argument('--scenter', type=str, default="0,0",
                        help='the xy center of shift search, in pixel')
    parser.add_argument('--ang2', type=float, default=0.5,
                        help='rotation angle step of the second round')
    parser.add_argument('--shift2', type=float, default=0.5,
                        help='shift step of the second round')
    parser.add_argument('--addang', type=float, default=0,
                        help='add this angle to the final result, in degree')
    parser.add_argument('--sigma', type=float, default=2,
                        help='sigma of the soft edge when lowpass and mask, in pixel')
    
    args = parser.parse_args()
    main(args)
    