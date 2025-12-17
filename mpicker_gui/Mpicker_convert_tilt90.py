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
from scipy.spatial.transform import Rotation
import argparse
from mpicker_star import read_star_loop, write_star_loop


def matrix2angles(matrix):
    # use Relion convention
    Nx = matrix[:, 0, 2] # N is z axis
    Ny = matrix[:, 1, 2]
    Nz = matrix[:, 2, 2]
    Ux = matrix[:, 0, 0] # U is x axis
    Uy = matrix[:, 1, 0]
    Uz = matrix[:, 2, 0]
    Nz = np.clip(Nz, -1, 1) # avoid numerical error
    tilt = np.arccos(Nz)
    psi = np.arctan2(Ny, -Nx) # notice the positive direction of all angles in relion are clockwise
    rot = np.zeros_like(tilt)
    tanp = np.abs(np.tan(psi))
    # mask = tanp == 0
    # rot[mask] = np.arctan2(-Uy[mask], Ux[mask])
    mask = tanp <= 1
    p, t, uy, uz = psi[mask], tilt[mask], Uy[mask], Uz[mask]
    rot[mask] = np.arctan2( -( np.sin(p)*np.cos(t)*uz + np.sin(t)*uy ) / np.cos(p), uz)
    mask = tanp > 1
    p, t, ux, uz = psi[mask], tilt[mask], Ux[mask], Uz[mask]
    rot[mask] = np.arctan2( ( np.cos(p)*np.cos(t)*uz - np.sin(t)*ux ) / np.sin(p), uz)
    mask = np.isclose(Nz, 1)
    psi[mask] = 0
    tilt[mask] = 0
    rot[mask] = np.arctan2(-Uy[mask], Ux[mask])
    mask = np.isclose(Nz, -1)
    psi[mask] = 0
    tilt[mask] = np.pi
    rot[mask] = np.arctan2(-Uy[mask], -Ux[mask])
    return np.rad2deg(rot), np.rad2deg(tilt), np.rad2deg(psi)


def angles2matrix(rot, tilt, psi):
    # use Relion convention. angles can be numbers or 1d arrays
    return Rotation.from_euler('zyz', np.column_stack((-rot, -tilt, -psi)), degrees=True).as_matrix()


def main(args):
    fin = args.i
    fout = args.o
    inv = args.inv
    pixel = args.p
    sx, sy, sz = map(int, args.s.split(','))
    name = args.name
    skip_ang = args.skip_ang
    skip_ori = args.skip_ori
    rand_rot = args.rand_rot
    shiftz = args.shiftz

    if not inv:
        x, y, z, rot, tilt, psi = np.loadtxt(fin, unpack=True)
        if rand_rot:
            rot = np.random.uniform(-180, 180, len(rot))
        tiltprior = 90
        psiprior = 0
        matrix_prior = angles2matrix(0, tiltprior, psiprior)[0]
        matrix = angles2matrix(rot, tilt, psi)
        matrix_tomo = matrix @ matrix_prior.T
        rottomo, tilttomo, psitomo = matrix2angles(matrix_tomo)
        keys = ['rlnTomoSubtomogramRot', 'rlnTomoSubtomogramTilt', 'rlnTomoSubtomogramPsi', 
                'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'rlnAngleTiltPrior', 'rlnAnglePsiPrior', 
                'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst', 'rlnCenteredCoordinateZAngst']
        if name is not None:
            keys = ['rlnTomoName'] + keys
        x = (x - sx/2) * pixel
        y = (y - sy/2) * pixel
        z = (z - sz/2) * pixel

        movex, movey, movez = (matrix[:,:,2] * shiftz).T # matrix * (0,0,shiftz)
        x, y, z = x + movex, y + movey, z + movez

        data = []
        for xx, yy, zz, r, t, p in zip(x, y, z, rottomo, tilttomo, psitomo):
            xx, yy, zz, r, t, p, r_p, t_p, p_p = [f"{value:7.2f}" for value in 
                                           [xx, yy, zz, r, t, p, 0, tiltprior, psiprior]]
            line = [r, t, p, r_p, t_p, p_p, t_p, p_p, xx, yy, zz]
            if name is not None:
                line = [name] + line
            data.append(line)
        write_star_loop(keys, data, fout, "particles", overwrite=True, is_dict=False)
        return
    
    else:
        keys, data = read_star_loop(fin, "particles")
        if name is not None:
            data = [d for d in data if d['rlnTomoName']==name]
        x = np.array([float(d['rlnCenteredCoordinateXAngst']) for d in data])
        y = np.array([float(d['rlnCenteredCoordinateYAngst']) for d in data])
        z = np.array([float(d['rlnCenteredCoordinateZAngst']) for d in data])
        if not skip_ori:
            ox, oy, oz = [np.zeros_like(x) for _ in range(3)]
            if "rlnOriginXAngst" in keys:
                ox = np.array([float(d['rlnOriginXAngst']) for d in data])
            if "rlnOriginYAngst" in keys:
                oy = np.array([float(d['rlnOriginYAngst']) for d in data])
            if "rlnOriginZAngst" in keys:
                oz = np.array([float(d['rlnOriginZAngst']) for d in data])
            shiftxyz = np.column_stack((-ox, -oy, -oz))

        rottomo, tilttomo, psitomo, rotprior, tiltprior, psiprior = [np.zeros_like(x) for _ in range(6)]
        if "rlnTomoSubtomogramRot" in keys:
            rottomo = np.array([float(d['rlnTomoSubtomogramRot']) for d in data])
        if "rlnTomoSubtomogramTilt" in keys:
            tilttomo = np.array([float(d['rlnTomoSubtomogramTilt']) for d in data])
        if "rlnTomoSubtomogramPsi" in keys:
            psitomo = np.array([float(d['rlnTomoSubtomogramPsi']) for d in data])
        if skip_ang:
            if "rlnAngleRotPrior" in keys:
                rotprior = np.array([float(d['rlnAngleRotPrior']) for d in data])
            if "rlnAngleTiltPrior" in keys:
                tiltprior = np.array([float(d['rlnAngleTiltPrior']) for d in data])
            if "rlnAnglePsiPrior" in keys:
                psiprior = np.array([float(d['rlnAnglePsiPrior']) for d in data])
        else:
            if "rlnAngleRot" in keys:
                rotprior = np.array([float(d['rlnAngleRot']) for d in data])
            if "rlnAngleTilt" in keys:
                tiltprior = np.array([float(d['rlnAngleTilt']) for d in data])
            if "rlnAnglePsi" in keys:
                psiprior = np.array([float(d['rlnAnglePsi']) for d in data])

        matrix_tomo = angles2matrix(rottomo, tilttomo, psitomo)
        matrix_prior = angles2matrix(rotprior, tiltprior, psiprior)      
        matrix = matrix_tomo @ matrix_prior
        rot, tilt, psi = matrix2angles(matrix)

        if not skip_ori:
            shx, shy, shz = np.einsum('nij,nj->in', matrix_tomo, shiftxyz)
            x, y, z = x + shx, y + shy, z + shz

        movex, movey, movez = (matrix[:,:,2] * shiftz).T
        x, y, z = x + movex, y + movey, z + movez

        x, y, z = x/pixel+sx/2, y/pixel+sy/2, z/pixel+sz/2
        if rand_rot:
            rot = np.random.uniform(-180, 180, len(rot))
        result = np.column_stack((x, y, z, rot, tilt, psi))
        np.savetxt(fout, result, fmt='%7.2f')
        return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert Euler angles if you hope TiltPrior=90 (membrane perpendicular to X axis, as Relion5)")
    parser.add_argument('--i', type=str, required=True,
                        help='input file with 6 columns, x y z rot tilt psi, xyz in pixels')
    parser.add_argument('--o', type=str, required=True,
                        help='output star file (Relion5 format)')
    parser.add_argument('--s', type=str, required=True,
                        help='xyz size of the tomo, such as "600,600,200", for the rlnCenteredCoordinate[XYZ]Angst')
    parser.add_argument('--p', type=float, required=True,
                        help='pixel size of the tomo, in Angstrom, for the rlnCenteredCoordinate[XYZ]Angst')
    parser.add_argument('--name', type=str,
                        help='optional. when not --inv, you can provide rlnTomoName. when --inv, you can specify which rlnTomoName to convert. ')
    parser.add_argument('--inv', action='store_true',
                        help='convert back. provide the star file and output 6 columns (x y z rot tilt psi)')
    parser.add_argument('--skip_ang', action='store_true',
                        help='only useful when --inv, will use rlnAngleXXXPrior instead of rlnAngleXXX')
    parser.add_argument('--skip_ori', action='store_true',
                        help='only useful when --inv, will ignore rlnOrigin[XYZ]Angst, just use rlnCenteredCoordinate[XYZ]Angst')
    parser.add_argument('--rand_rot', action='store_true',
                        help='randomize the rot angle of input at first (when not --inv), or randomize the rot of output at last (when --inv)')
    parser.add_argument('--shiftz', type=float, default=0,
                        help='recenter by shift the center along z axis, in Angstrom. default 0.')
    args = parser.parse_args()
    main(args)