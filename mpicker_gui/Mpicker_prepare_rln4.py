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
import argparse
import os
from mpicker_star import read_star_loop, write_star_loop
from Mpicker_prepare_rln2 import load_data, read_name_tomo, read_mpicker_out2, shift_coords
from Mpicker_convert_tilt90 import matrix2angles, angles2matrix


def main(args):
    fin = args.i
    fout = args.o
    link = args.link
    scale = args.s
    from_rln2 = args.from_rln2
    tilt90 = args.tilt90
    shiftz = args.shiftz
    no_prior_rot = args.no_prior_rot
    no_prior_tilt = args.no_prior_tilt
    no_prior_psi = args.no_prior_psi
    center_str = args.center

    if from_rln2:
        keys, data = read_star_loop(fin)
        origin = "rlnOriginX" in keys and "rlnOriginY" in keys and "rlnOriginZ" in keys
        if not origin:
            print("rlnOrigin[XYZ] not found in the input star file, ignore.")
        if "rlnMicrographName" not in keys:
            raise Exception("rlnMicrographName not found in the input star file.")
        coords, angles = load_data(data, not origin, True) # relion will floor the coords when extract particles
        tomos = [os.path.splitext(os.path.basename(d['rlnMicrographName']))[0] for d in data]
        tomo_list = []
        for tomo in tomos:
            if tomo not in tomo_list:
                tomo_list.append(tomo)
    else:
        data = read_mpicker_out2(fin)
        if link is not None:
            dict_name_tomo = read_name_tomo(link)
        else:
            dict_name_tomo = {d['name']:d['name'] for d in data}
        coords, angles = load_data(data)
        tomos = [dict_name_tomo[d['name']] for d in data]
        tomo_list = list(dict_name_tomo.values())

    rot, tilt, psi = angles.T
    if shiftz != 0:
        coords = shift_coords(coords, shiftz, tilt, psi)
    coords *= scale
    if center_str is not None:
        sx,sy,sz,pixel = map(float, center_str.split(',')[0:4])
        coords = (coords - np.array([sx/2,sy/2,sz/2])) * pixel

    if tilt90:
        rotprior = 0.
        tiltprior = 90.
        psiprior = 0.
        matrix_prior = angles2matrix(rotprior, tiltprior, psiprior)[0]
        matrix = angles2matrix(rot, tilt, psi)
        matrix_tomo = matrix @ matrix_prior.T
        rottomo, tilttomo, psitomo = matrix2angles(matrix_tomo)
        rot = 0*rot + rotprior
        tilt = 0*tilt + tiltprior
        psi = 0*psi + psiprior

    data = [coords]
    if center_str is not None:
        keys = ['rlnTomoName', 'rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst', 'rlnCenteredCoordinateZAngst']
    else:
        keys = ['rlnTomoName', 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    if tilt90:
        keys += ['rlnTomoSubtomogramRot', 'rlnTomoSubtomogramTilt', 'rlnTomoSubtomogramPsi', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
        data += [rottomo, tilttomo, psitomo, rot, tilt, psi]
    else:
        keys += ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
        data += [rot, tilt, psi]
    if not no_prior_rot:
        keys.append('rlnAngleRotPrior')
        data.append(rot)
    if not no_prior_tilt:
        keys.append('rlnAngleTiltPrior')
        data.append(tilt)
    if not no_prior_psi:
        keys.append('rlnAnglePsiPrior')
        data.append(psi)
    data = np.column_stack(data)

    result = []
    for tomo in tomo_list:
        mask = [t==tomo for t in tomos]
        result += [ [tomo]+[f"{value:7.2f}" for value in line] for line in data[mask] ]
    write_star_loop(keys, result, fout, "particles", overwrite=True, is_dict=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert the result (out2) of Mpicker_convert_class2d.py to particles file required by Relion4 for STA.")
    parser.add_argument('--i', type=str, required=True,
                        help='the out2 file from Mpicker_convert_class2d, can be star file or plain text')
    parser.add_argument('--link', '--l', type=str,
                        help='file contains 2 column, 1st is the projections name in --i file, 2nd is the output tomogram name (no suffix) for it.')
    parser.add_argument('--o', type=str, required=True,
                        help='output star file')
    parser.add_argument('--s', type=float, default=1,
                        help='multiply the coordinates by this number (default 1), so that they correspond to bin1 tomogram')
    parser.add_argument('--from_rln2', action='store_true',
                        help='if True, the --i should be _data.star file of Relion2, and then will output particles file for Relion4')
    parser.add_argument('--tilt90', action='store_true',
                        help='if True, the pseudo-subtomo will have orientations, so that rlnAngleTiltPrior=90.')
    parser.add_argument('--shiftz', type=float, default=0,
                        help='shift the volume center along z axis, in pixel (before multiply --s), optional')
    parser.add_argument('--no_prior_rot', action='store_true',
                        help='if True, will not write rlnAngleRotPrior in output file')
    parser.add_argument('--no_prior_tilt', action='store_true',
                        help='if True, will not write rlnAngleTiltPrior in output file')
    parser.add_argument('--no_prior_psi', action='store_true',
                        help='if True, will not write rlnAnglePsiPrior in output file')
    parser.add_argument('--center', type=str, default=None,
                        help='optional, provide bin1 tomogram size and pixel size here, such as "4000,4000,1000,1.35", \
                              then will use rlnCenteredCoordinate[XYZ]Angst in output file')
    args = parser.parse_args()
    main(args)
    