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
from mpicker_star import read_star_loop, write_star_loop, read_list
import mrcfile



def prepare_all(tomos_pre, dir_out, Voltage, Cs, AmpContrast, PixelSize, UseOnlyLowerTiltDefociLimit, Bfactor):
    tomos_dir = [os.path.join(dir_out, f"Tomograms/{tomo}") for tomo in tomos_pre]
    ctfs_dir = [os.path.join(dir_out, f"Particles/Tomograms/{tomo}") for tomo in tomos_pre]
    f_all_tomograms = os.path.join(dir_out, "all_tomograms.star")
    if os.path.isfile(f_all_tomograms):
        print("all_tomograms.star already exists, skip gerating it")
    else:
        all_tomograms = [[f"./Tomograms/{tomo}/{tomo}.mrc"] for tomo in tomos_pre]
        write_star_loop(["rlnMicrographName"], all_tomograms, f_all_tomograms, overwrite=True, is_dict=False)
    
    starhead_U=[
    "data_images\n",
    "loop_\n",
    "_rlnDefocusU #1\n",
    "_rlnVoltage #2\n",
    "_rlnSphericalAberration #3\n",
    "_rlnAmplitudeContrast #4\n",
    "_rlnAngleRot #5\n",
    "_rlnAngleTilt #6\n",
    "_rlnAnglePsi #7\n",
    "_rlnCtfBfactor #8\n",
    "_rlnCtfScalefactor #9\n"
    ]

    starhead_UV=[
    "data_images\n",
    "loop_\n",
    "_rlnDefocusU #1\n",
    "_rlnDefocusV #2\n",
    "_rlnDefocusAngle #3\n",
    "_rlnVoltage #4\n",
    "_rlnSphericalAberration #5\n",
    "_rlnAmplitudeContrast #6\n",
    "_rlnAngleRot #7\n",
    "_rlnAngleTilt #8\n",
    "_rlnAnglePsi #9\n",
    "_rlnCtfBfactor #10\n",
    "_rlnCtfScalefactor #11\n"
    ]

    ctf_script = """#!/bin/bash
if [ "$#" -ne 2 ]
then echo "Usage: $0 <box_size> <threads>"
exit 1
fi
maxthread=$2
file=do_all_reconstruct_ctfs.sh

tmp_fifo=$$.fifo
mkfifo $tmp_fifo
exec 6<>$tmp_fifo
rm $tmp_fifo
for ((i=0;i<$maxthread;i++));do
    echo >&6
done

while read line
do read -u6
{
eval $line
echo >&6
}&
done < $file

wait
exec 6>&-
"""

    def write_particle_defocus(out_pre, xyz, tilts, doses, defocusU, defocusV=None, defocusAng=None):
        x, z = xyz[0] * PixelSize, xyz[2] * PixelSize
        d_defocus = x*np.sin(tilts/180*np.pi) - z*np.cos(tilts/180*np.pi)
        defocusU = defocusU + d_defocus # not defocus_avg + sin(tilt/180*pi)*(xx*cos(tilt/180*pi)+zz*sin(tilt/180*pi))
        if defocusV is not None:
            defocusV = defocusV + d_defocus
            content = ["%-10.2f %10.2f %8.2f %5.1f %5.2f %5.2f 0.00 %8.2f 0.00 %8.2f %5.2f \n"%(focu,focv,ang,Voltage,Cs,AmpContrast,tilt,Bfactor*dose,np.cos(tilt/180*np.pi))
                    for focu,focv,ang,tilt,dose in zip(defocusU,defocusV,defocusAng,tilts,doses)]
            content = starhead_UV + content
        else:
            content = ["%-10.2f %5.1f %5.2f %5.2f 0.00 %8.2f 0.00 %8.2f %5.2f \n"%(focu,Voltage,Cs,AmpContrast,tilt,Bfactor*dose,np.cos(tilt/180*np.pi))
                    for focu,tilt,dose in zip(defocusU,tilts,doses)]
            content = starhead_U + content
        with open(os.path.join(dir_out, out_pre+".star"), 'w') as f:
            f.writelines(content)
        cmd_line = 'relion_reconstruct --i ' + out_pre+".star" + ' --o ' + out_pre+".mrc" + ' --reconstruct_ctf $1 --angpix ' + str(PixelSize) + '\n'
        return cmd_line

    ctf_cmd=[]
    for tomo_pre,tomo_dir,ctf_dir in zip(tomos_pre,tomos_dir,ctfs_dir):
        print(tomo_pre)
        orderfile = tomo_dir + "/" + tomo_pre + ".order"
        defocusfile = tomo_dir + "/" + tomo_pre + ".defocus"
        coordfile = tomo_dir + "/" + tomo_pre + ".coords"
        tomofile = tomo_dir + "/" + tomo_pre + ".mrc"
        os.makedirs(ctf_dir, exist_ok=True)

        orders = np.loadtxt(orderfile,ndmin=2,dtype=float)[:,:2]
        defocus = np.loadtxt(defocusfile,ndmin=2,dtype=float)
        coords = np.loadtxt(coordfile,ndmin=2,dtype=float)[:,:3]
        if len(orders) != len(defocus):
            print(f"Error: the line number of .order and .defocus is not consistent, skip it.")
            continue
        tilts, doses = orders.T
        x, y, z = coords.T
        with mrcfile.open(tomofile, permissive=True, header_only=True) as mrc:
            sx, sy, sz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
        x,y,z = x-sx/2,y-sy/2,z-sz/2
        if defocus.shape[1] >= 3:
            defocusU, defocusV, defocusAng = defocus.T[0:3]
        else:
            defocusU = defocus.T[0]
            defocusV, defocusAng = None, None
        mask_low = np.abs(tilts) <= UseOnlyLowerTiltDefociLimit
        mask_high = np.abs(tilts) > UseOnlyLowerTiltDefociLimit
        defocusU[mask_high] = defocusU[mask_low].mean()
          
        idx=0
        for dx,dy,dz in zip(x,y,z):
            idx+=1
            out_pre = f"Particles/./Tomograms/{tomo_pre}/{tomo_pre}_ctf{idx:06d}"
            cmd_line = write_particle_defocus(out_pre, (dx, dy, dz), tilts, doses, defocusU, defocusV, defocusAng)
            ctf_cmd.append(cmd_line)

    fcmd = os.path.join(dir_out, "do_all_reconstruct_ctfs.sh")
    with open(fcmd,'w') as f:
        f.writelines(ctf_cmd)
    os.chmod(fcmd, 0o755)
    fscript = os.path.join(dir_out, "para_3dctf.sh")
    with open(fscript,'w') as f:
        f.write(ctf_script)
    os.chmod(fscript, 0o755)
    return


def shift_coords(coords:np.ndarray, shiftz:float, tilt:np.ndarray, psi:np.ndarray) -> np.ndarray:
    tilt = -np.deg2rad(tilt)
    psi = -np.deg2rad(psi)
    dz = np.cos(tilt)
    dy = np.sin(tilt) * np.sin(psi)
    dx = np.sin(tilt) * np.cos(psi)
    shiftz = shiftz * np.column_stack((dx, dy, dz))
    return coords + shiftz


def load_data(data, no_origin=False, floor_coord=False):
    x = []
    y = []
    z = []
    rot = []
    tilt = []
    psi = []
    ox = []
    oy = []
    oz = []
    for d in data:
        x.append(float(d["rlnCoordinateX"]))
        y.append(float(d["rlnCoordinateY"]))
        z.append(float(d["rlnCoordinateZ"]))
        rot.append(float(d["rlnAngleRot"]))
        tilt.append(float(d["rlnAngleTilt"]))
        psi.append(float(d["rlnAnglePsi"]))
        if not no_origin:
            ox.append(float(d["rlnOriginX"]))
            oy.append(float(d["rlnOriginY"]))
            oz.append(float(d["rlnOriginZ"]))
    coords = np.column_stack((x, y, z))
    angles = np.column_stack((rot, tilt, psi))
    if floor_coord:
        print("will floor the rlnCoordinate at first.")
        coords = np.floor(coords)
    if not no_origin:
        coords -= np.column_stack((ox, oy, oz))
    return coords, angles


def read_name_tomo(flink):
    dict_name_tomo = dict()
    with open(flink, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            line = line.split()
            if len(line) < 2:
                continue
            name, tomo = line[0:2]
            dict_name_tomo[name] = tomo
    return dict_name_tomo


def read_mpicker_out2(fin):
    _, data = read_star_loop(fin)
    if len(data) == 0:
        data = read_list(fin, keys=["name", "idx", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginX", "rlnOriginY", "rlnOriginZ"])
        if len(data) == 0:
            raise Exception("no data found in input file.")
    return data


def main(args):
    fin = args.i
    dout = args.o
    link = args.link
    scale = args.s
    pixel = args.p
    shiftz = args.shiftz
    no_round = args.no_round
    f_all = args.prepare_all
    if not os.path.isdir(dout):
        os.makedirs(dout)

    data = read_mpicker_out2(fin)
    if link is not None:
        dict_name_tomo = read_name_tomo(link)
    else:
        dict_name_tomo = {d["name"]: d["name"] for d in data}
        
    dict_name_data = dict()
    for name in dict_name_tomo.keys():
        dict_name_data[name] = []
    for d in data:
        name = d["name"]
        if name in dict_name_tomo:
            dict_name_data[name].append(d)
    
    results = []
    keys = ["rlnMicrographName", "rlnImageName", "rlnCtfImage","rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
    if pixel is not None:
        keys += ["rlnMagnification", "rlnDetectorPixelSize"]
    for name, tomo in dict_name_tomo.items():
        print(tomo)
        coords, angles = load_data(dict_name_data[name])
        if shiftz != 0:
            coords = shift_coords(coords, shiftz, angles[:,1], angles[:,2])
        coords *= scale
        fcoords = os.path.join(dout, f"{tomo}.coords")
        if not no_round:
            coords = np.round(coords)
        np.savetxt(fcoords, coords, fmt="%7.2f")
        for i in range(len(coords)):
            x, y, z = coords[i]
            r, t, p = angles[i]
            idx = i + 1
            Mname = f"./Tomograms/{tomo}/{tomo}.mrc"
            Iname = f"Extract/extract_tomo/./Tomograms/{tomo}/{tomo}{idx:06d}.mrc"
            Cname = f"Particles/./Tomograms/{tomo}/{tomo}_ctf{idx:06d}.mrc"
            line = [Mname, Iname, Cname, f"{x:7.2f}", f"{y:7.2f}", f"{z:7.2f}", f"{r:7.2f}", f"{t:7.2f}", f"{p:7.2f}"]
            if pixel is not None:
                line += ["10000", f"{pixel:.2f}"]
            results.append(line)
    write_star_loop(keys, results, os.path.join(dout, "particles.star"), overwrite=True, is_dict=False)

    if f_all is not None:
        print("Prepare all...")
        lines = []
        with open(f_all, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                line = line.split()
                if len(line) > 0:
                    lines.append(float(line[0]))
        Voltage, Cs, AmpContrast, PixelSize, UseOnlyLowerTiltDefociLimit, Bfactor = lines
        tomos_pre = list(dict_name_tomo.values())
        for tomo in tomos_pre:
            tomo_dir = os.path.join(dout, f"Tomograms/{tomo}")
            if not os.path.isdir(tomo_dir):
                raise Exception(f"{tomo_dir} does not exist.")
            os.rename(os.path.join(dout, f"{tomo}.coords"), os.path.join(tomo_dir, f"{tomo}.coords"))
        prepare_all(tomos_pre, dout, Voltage, Cs, AmpContrast, PixelSize, UseOnlyLowerTiltDefociLimit, Bfactor)
        print("Now you can generate 3dCTFs using do_all_reconstruct_ctfs.sh in the working directory, or you can use:")
        print(f"./para_3dctf.sh <box_size> <threads>")
        print("Then you can extract the particles in Relion, and use the particles.star directly for reconstruction or refinement.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert the result (out2) of Mpicker_convert_class2d.py to files required by Relion2 for STA.")
    parser.add_argument('--i', type=str, required=True,
                        help='the out2 file from Mpicker_convert_class2d, can be star file or plain text')
    parser.add_argument('--link', '--l', type=str, default=None,
                        help='file contains 2 column, 1st is the projections name in --i file, 2nd is the output tomogram name (no suffix) for it. \
                              if not provided, the output name will be the same as the name in --i file.')
    parser.add_argument('--o', type=str, required=True,
                        help='output directory. will output a coords file for each tomogram and one particles.star')
    parser.add_argument('--s', type=float, default=1,
                        help='multiply the coordinates by this number, default 1')
    parser.add_argument('--p', type=float, default=None,
                        help='can write pixel size to the output star file, optional')
    parser.add_argument('--shiftz', type=float, default=0,
                        help='shift the volume center along z axis, in pixel (before multiply --s), optional')
    parser.add_argument('--no_round', action='store_true',
                        help='if True, will not round the coordinates to integer at last. notice that Relion will floor the coords when extract.')
    parser.add_argument('--prepare_all', type=str, default=None,
                        help='provide a parameters file to prepare all files for Relion2, optional. then the --o is the working directoy. \
                              The parameters file should contain 6 lines in order: \
                              Voltage(in KV), Cs(in mm), AmpContrast, PixelSize(in A), UseOnlyLowerTiltDefociLimit(in degree), Bfactor. \
                              The tomoxx.mrc tomoxx.order and tomoxx.defocus should exist in working_directoy/Tomograms/tomoxx/ for each tomogram. \
                              tomoxx.defocus is the defocus(in A, one column) for each tilt. The other files are the same as the relion_prepare_subtomograms')
    args = parser.parse_args()
    main(args)
