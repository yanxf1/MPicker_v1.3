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

import mrcfile
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Deque
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion
from collections import deque
from time import time
from tqdm import tqdm
import configparser, argparse, os
np.random.seed(42)

try:
    print("Importing Cython module...")
    import pyximport
    pyximport.install(setup_args={"include_dirs":[np.get_include()]}, language_level=3)
    from mpicker_autoextract_cy import region_grow_cython, region_grow_2d_cython
    use_cython = True
except Exception as e:
    print(e)
    print("Warning: Failed to import Cython module, using SLOWER Python implementation.")
    print("You can install it by: conda install cython")
    use_cython = False

def region_grow_python(
    normals: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    cosines_all: np.ndarray,
    sorted_indices: np.ndarray,
    cos_thres: float,
    region_cos_thres: float,
    visited: np.ndarray=None
) -> List[List[int]]:
    assert normals.ndim == 2 and len(normals) > 0
    assert len(sorted_indices) == len(normals)

    if visited is None:
        visited = np.zeros(len(normals), dtype=bool)
    else:
        visited = visited.astype(bool)
        assert len(visited) == len(normals)
    regions: List[List[int]] = []
    regions_normal: List[List[float]] = []

    bar = tqdm(total=len(visited), desc="Region Growing Python")
    for idx in sorted_indices:
        if visited[idx]:
            continue
        region_indices: List[int] = []
        queue: Deque[int] = deque([idx])
        region_normal_sum = np.zeros(3, dtype=np.float32)

        while queue:
            current = queue.popleft()
            if visited[current]:
                continue
            visited[current] = True
            region_indices.append(int(current))
            if np.dot(region_normal_sum, normals[current]) > 0:
                region_normal_sum += normals[current]
            else:
                region_normal_sum -= normals[current]
            region_normal = region_normal_sum / np.linalg.norm(region_normal_sum)
            neighbors = indices[indptr[current]:indptr[current+1]]
            cosines = cosines_all[indptr[current]:indptr[current+1]]
            for nbr, cos in zip(neighbors, cosines):
                if visited[nbr]:
                    continue
                if cos < cos_thres:
                    continue
                region_cos = np.dot(region_normal, normals[nbr])
                if abs(region_cos) < region_cos_thres:
                    continue
                queue.append(nbr)

        regions.append(region_indices)
        region_normal = region_normal_sum / np.linalg.norm(region_normal_sum)
        regions_normal.append(region_normal.tolist())
        bar.update(len(region_indices))
    bar.update(bar.total - bar.n)
    bar.close()
    return regions, regions_normal


def smooth_point(points, sigma=1, k=10):
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k, workers=8)
    neighbors = points[indices]
    weight = np.exp(-distances**2 / (2 * sigma**2))
    weight /= weight.sum(axis=1, keepdims=True)
    smoothed_points = (weight[..., np.newaxis] * neighbors).sum(axis=1)
    return smoothed_points


def plot_regions(points, regions, num):
    if num <= 0:
        return
    for i in range(min(num, len(regions))):
        r = regions[i]
        print(f"Region {i}: {len(r)} points.")
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(points[r])
        p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
        p.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([p], width=700, height=700)


def plot_regions_all(points, regions):
    if len(regions) == 0:
        return
    ps = o3d.geometry.PointCloud()
    for r in regions:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(points[r])
        p.paint_uniform_color(np.random.rand(3))
        ps += p
    ps.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    o3d.visualization.draw_geometries([ps], width=700, height=700)


def region_grow_3d(
    pcd: o3d.geometry.PointCloud,
    knn: int,
    rbound: float,
    angle_thres: float,
    region_angle_thres: float,
    sorted_indices: List[int]=None,
    sorted_number: int=0
) -> Tuple[List[List[int]], List[List[float]]]:
    t0 = time()
    cos_thres = np.cos(np.deg2rad(angle_thres))
    region_cos_thres = np.cos(np.deg2rad(region_angle_thres))

    print("Computing normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    print("Building adjacency...")
    tree = KDTree(points)
    _, idx = tree.query(points, k=knn, distance_upper_bound=rbound, workers=8)
    # calculate cosine between neighbors
    mask = idx < len(points)
    neighbor_count = mask.sum(axis=1)
    indptr = np.insert(neighbor_count.cumsum(), 0, 0)
    indices = idx[mask]
    data = np.ones_like(indices)
    adj = csr_matrix((data, indices, indptr), shape=(len(points), len(points))).tocoo()
    rows, cols = adj.row, adj.col
    cosines = np.einsum('ij,ij->i', normals[rows], normals[cols])
    cosines = csr_matrix((cosines, (rows, cols)), shape=(len(points), len(points)))
    indptr = cosines.indptr
    indices = cosines.indices
    cos_all = np.abs(cosines.data)
    print(f"{time()-t0:.1f}s.")

    indptr = indptr.astype(np.int32)
    indices = indices.astype(np.int32)
    normals = np.asarray(pcd.normals).astype(np.float32)
    cos_all = cos_all.astype(np.float32)
    skip_mask = None
    if sorted_indices is None:
        sorted_idx = np.random.permutation(len(points)).astype(np.int32)
    else:
        assert len(sorted_indices) == len(points)
        sorted_idx = sorted_indices.astype(np.int32)
        if sorted_number > 0:
            skip_mask = np.ones(len(points), dtype=np.uint8)
            skip_mask[sorted_indices[:sorted_number]] = 0
    if use_cython:
        print("Region growing Cython...")
        t0 = time()
        regions, regions_nomal = region_grow_cython(normals, indptr, indices, cos_all, sorted_idx, cos_thres, region_cos_thres, skip_mask)
        print(f"{time()-t0:.1f}s.")
    else:
        regions, regions_nomal = region_grow_python(normals, indptr, indices, cos_all, sorted_idx, cos_thres, region_cos_thres)
    return regions, regions_nomal


def pair2csr(row, col, data, length, fill_self=None):
    if fill_self is not None:
        assert len(fill_self) == length
        fill = np.arange(length, dtype=row.dtype)
        fill_data = fill_self.astype(data.dtype)
        row, col = np.concatenate([row, col, fill]), np.concatenate([col, row, fill])
        data = np.concatenate([data, data, fill_data])
    else:
        row, col = np.concatenate([row, col]), np.concatenate([col, row])
        data = np.concatenate([data, data])
    matrix = csr_matrix((data, (row, col)), shape=(length, length))
    return matrix


def prepare_norm2z(points, normalxyz, rbound):
    n_vector = normalxyz / np.linalg.norm(normalxyz)
    theta = np.arccos(n_vector[2])
    phi = np.arctan2(n_vector[1], n_vector[0])
    rot_matrix = Rotation.from_euler('zy', [-phi, -theta]).as_matrix()
    points = points @ rot_matrix.T
    N = len(points)
    # smooth z
    tree = KDTree(points)
    row, col = tree.query_pairs(rbound, output_type="ndarray").T
    dist3d = np.linalg.norm(points[row] - points[col], axis=1)
    weight = np.exp(-dist3d**2 / (2 * rbound**2))
    weight = pair2csr(row, col, weight, N, np.ones(N))
    points[:, 2] = weight @ points[:, 2] / weight.sum(axis=1).A1
    # neighbors in xy
    z = points[:, 2]
    points2d = points[:, :2]
    tree = KDTree(points2d)
    row, col = tree.query_pairs(rbound, output_type="ndarray").T
    z_diff = np.abs(z[row] - z[col])
    dist2d = np.linalg.norm(points2d[row] - points2d[col], axis=1) + 1e-12
    z_diff = pair2csr(row, col, z_diff, N)
    dist2d = pair2csr(row, col, dist2d, N)
    z_diff, indptr, indices = z_diff.data, z_diff.indptr, z_diff.indices
    dist2d = dist2d.data
    return z_diff, dist2d, indptr, indices


def region_grow_2d_python(zdiff_data, dist2d_data, indptr, indices, 
                          sorted_indices, tan_near, tan_far, dist_compete):
    N = len(sorted_indices)
    assert N == len(indptr) - 1
    visited = np.zeros(N, dtype=bool)
    regions: List[np.ndarray] = []
    for idx in sorted_indices:
        if visited[idx]:
            continue
        visited_this = visited.copy()
        region_indices: List[int] = []
        queue: Deque[Tuple[int, bool]] = deque([(idx, True)])
        far_points = np.zeros(N, dtype=bool)
        while queue:
            current, is_this = queue.popleft()
            if visited_this[current]:
                continue
            if is_this:
                if far_points[current]:
                    continue
                visited_this[current] = True
                region_indices.append(int(current))
                zdiff = zdiff_data[indptr[current]:indptr[current+1]]
                dist2d = dist2d_data[indptr[current]:indptr[current+1]]
                tan = zdiff / dist2d
                neighbors = indices[indptr[current]:indptr[current+1]]
                next_data_this = neighbors[tan <= tan_near]
                next_data_this = [(n, True) for n in next_data_this]
                queue.extend(next_data_this)
                next_data_other = neighbors[tan > tan_far]
                far_points[next_data_other] = True
                next_data_other = [(n, False) for n in next_data_other]
                queue.extend(next_data_other)
            else:
                visited_this[current] = True
                zdiff = zdiff_data[indptr[current]:indptr[current+1]]
                dist2d = dist2d_data[indptr[current]:indptr[current+1]]
                tan = zdiff / dist2d
                neighbors = indices[indptr[current]:indptr[current+1]]
                mask = (tan <= tan_near) & (dist2d <= dist_compete)
                next_data = neighbors[mask] # near to current point
                far_points[next_data] = True
                next_data = [(n, False) for n in next_data]
                queue.extend(next_data)
        regions.append(region_indices)
        visited[region_indices] = True
    return regions


def expand_points(pointsxyz, shapexyz, expand):
    sx, sy, sz = shapexyz
    points = pointsxyz.astype(int)
    if expand:
        points[:, 0] = np.clip(points[:, 0], 0, sx-2)
        points[:, 1] = np.clip(points[:, 1], 0, sy-2)
        points[:, 2] = np.clip(points[:, 2], 0, sz-2)
        points = np.vstack([points,
                            points + np.array([1, 0, 0]),
                            points + np.array([0, 1, 0]),
                            points + np.array([0, 0, 1]),
                            points + np.array([1, 1, 0]),
                            points + np.array([1, 0, 1]),
                            points + np.array([0, 1, 1]),
                            points + np.array([1, 1, 1])])
    else:
        points[:, 0] = np.clip(points[:, 0], 0, sx-1)
        points[:, 1] = np.clip(points[:, 1], 0, sy-1)
        points[:, 2] = np.clip(points[:, 2], 0, sz-1)
    points = np.unique(points, axis=0)
    return points


def save2mpicker(pointsxyz, zyxshape, idx, fconfig, expand, flatten_parser=None):
    points = expand_points(pointsxyz, zyxshape[::-1], expand)
    config = configparser.ConfigParser()
    config['Parameter'] = {}
    config['Parameter']['id'] = str(idx)
    config['Parameter']['points'] = str([(points[0]+1).tolist()])
    config['Parameter']['mode'] = str(['simple'])
    config['Parameter']['facexyz'] = str(['x'])
    config['Parameter']['nearero'] = str(6)
    config['Parameter']['directionl2r'] = str(['Left To Right'])
    config['Parameter']['minsurf'] = str(10)
    config['Parameter']['ncpu'] = str(1)
    config['Parameter']['maxpixel'] = str(200)
    
    folder, fname = os.path.split(fconfig)
    fname = os.path.splitext(fname)[0] # tomo name
    folder = os.path.join(folder, fname) # tomo folder
    folder_surface = os.path.join(folder, f"surface_{idx}_{fname}")
    assert os.path.isdir(folder)
    if os.path.isdir(folder_surface):
        print(f"Warning: folder {folder_surface} exists, overwriting...")
    else:
        os.makedirs(folder_surface)
    fconfig_surface = os.path.join(folder_surface, f"surface_{idx}.config")
    fnpz = os.path.join(folder_surface, f"surface_{idx}_surf.mrc.npz")
    with open(fconfig_surface, 'w') as f:
        config.write(f)

    np.savez(fnpz, size_zyx=zyxshape, coords=points[:, ::-1])

    config_main = configparser.ConfigParser()
    config_main.read(fconfig, encoding='utf-8')
    assert config_main.has_section("Path")
    Surface_string = config_main.get('Path', 'Surface', fallback='None')
    if Surface_string == 'None':
        config_main.set('Path', 'Surface', f"surface_{idx}_{fname}")
    else:
        same_flag = False
        Surface_string = config_main.get('Path', 'Surface')
        Surface_name = f"surface_{idx}_{fname}"
        for Surface in Surface_string.split():
            if Surface ==  Surface_name:
                same_flag = True
        if same_flag == False:
            Surface_string = " ".join(sorted( Surface_string.split(), key=lambda x: int(x.split("_")[1]) ))
            Surface_string = Surface_string + " " +  Surface_name
            config_main.set('Path', 'Surface', Surface_string)
    with open(fconfig, 'w') as f:
        config_main.write(f)

    if flatten_parser is not None:
        fflatten = os.path.join(folder_surface, f"surface_{idx}-1.config")
        with open(fflatten, 'w') as f:
            flatten_parser.write(f)
        return fflatten
    else:
        return None


def create_boundary_6(mask):
    struct_ero = np.zeros((3, 3, 3))
    struct_ero[(1, 0, 2, 1, 1, 1, 1), (1, 1, 1, 0, 2, 1, 1), (1, 1, 1, 1, 1, 0, 2)] = 1
    boundary = mask - binary_erosion(mask, structure=struct_ero).astype(np.uint8)
    boundary[[0,-1],:]=0
    boundary[:,[0,-1]]=0
    return boundary


def create_mpicker(fout, fraw, fmask):
    assert os.path.isdir(fout), f"Output folder {fout} does not exist!"
    fout = os.path.abspath(fout)
    fname = os.path.splitext(os.path.basename(fraw))[0] # tomo name
    folder = os.path.join(fout, fname) # tomo folder
    fconfig = os.path.join(fout, f"{fname}.config")
    assert not os.path.isfile(fconfig), f"Config file {fconfig} exists!"
    fboundary = os.path.join(folder, "my_boundary_6.mrc")
    config = configparser.ConfigParser()
    config['Path'] = {}
    config['Path']['inputraw'] = os.path.abspath(fraw)
    config['Path']['inputmask'] = os.path.abspath(fmask)
    config['Path']['inputboundary'] = fboundary
    os.makedirs(folder, exist_ok=True)
    with mrcfile.open(fmask, permissive=True) as mrc:
        mask = mrc.data > mrc.data.mean()
        voxel = mrc.voxel_size
        boundary = create_boundary_6(mask)
    with mrcfile.new(fboundary, overwrite=True) as mrc:
        mrc.set_data(boundary)
        mrc.voxel_size = voxel
    with open(fconfig, 'w') as f:
        config.write(f)
    return fconfig


def load_mrc2point(fname, iter):
    with mrcfile.open(fname) as mrc:
        mask = mrc.data > mrc.data.mean()
        data_shape = mrc.data.shape
    if iter > 0:
        num_before = mask.sum()
        struct_ero = np.zeros((3, 3, 3))
        struct_ero[(1, 0, 2, 1, 1, 1, 1), (1, 1, 1, 0, 2, 1, 1), (1, 1, 1, 1, 1, 0, 2)] = 1
        mask = binary_erosion(mask, structure=struct_ero, iterations=iter)
        print(f"Erosion {iter}: {num_before} to {mask.sum()} points.")
    point_xyz = np.column_stack(np.nonzero(mask)[::-1])
    return point_xyz, data_shape


def flatten_config_to_mrc(config):
    parser = configparser.ConfigParser()
    parser.read(config, encoding='utf-8')
    method = parser.get('Parameter', 'Method')
    rbf_dist = parser.getint('Parameter', 'RBFSample')
    order = parser.getint('Parameter', 'PolyOrder')
    thick = parser.getint('Parameter', 'Thickness')
    prefix = os.path.splitext(config)[0]
    if method == 'RBF':
        num = rbf_dist
    else:
        num = order
    fmrc = f'{prefix}_{method}_{num}_thick_{thick}_result.mrc'
    return fmrc


def run_flatten(tomo_config, surf_configs):
    assert os.path.isfile(tomo_config), f"Tomo config {tomo_config} does not exist!"
    import sys, Mpicker_core_gui
    core_path = Mpicker_core_gui.__file__
    num_success = 0
    num_fail = 0
    tomo_list = []
    for i, surf_config in enumerate(surf_configs):
        print(f"{i+1}/{len(surf_configs)} ...")
        if surf_config is None or not os.path.isfile(surf_config):
            print(f"Surface config {surf_config} does not exist, skipping...")
            num_fail += 1
            continue
        cmd = f'{sys.executable} {core_path} --mode flatten --config_tomo {tomo_config} --config_surf {surf_config}'
        print(f'\033[0;35m{cmd}\033[0m')
        s = os.system(cmd)
        if s != 0:
            fname = os.path.basename(surf_config)
            print(f"{fname} failed, exit code: {s}")
            num_fail += 1
            continue
        fmrc = flatten_config_to_mrc(surf_config)
        if not os.path.isfile(fmrc):
            print(f"Flattened tomo {fmrc} not found, something wrong!")
            num_fail += 1
            continue
        num_success += 1
        tomo_list.append(fmrc)
    print(f"Flatten finished: {num_success} success, {num_fail} fail.")
    return tomo_list


def check_tomo_size(tomo_list):
    size = 0
    for tomo in tomo_list:
        with mrcfile.mmap(tomo, permissive=True) as mrc:
            sz, sy, sx = mrc.data.shape
        size = max(size, sx, sy)
    return size


def run_epicker_batch(tomo_list, epicker_model, max_num, output_id, pad, epicker_path, gpuid):
    import sys, Mpicker_epicker_batch
    py_path = Mpicker_epicker_batch.__file__
    if len(tomo_list) == 0:
        return
    tomo_dir = os.path.dirname(os.path.dirname(tomo_list[0]))
    tmp_dir = Mpicker_epicker_batch.make_tmpdir(tomo_dir)
    tmp_file = os.path.join(tmp_dir, "epicker_fin.txt")
    with open(tmp_file, 'w') as f:
        for tomo in tomo_list:
            f.write(f"{tomo} 1\n")
    if pad == 0:
        pad = max(1024, check_tomo_size(tomo_list))
    cmd = f'{sys.executable} {py_path} --model {epicker_model} --fin {tmp_file} --out {tmp_dir} --max_num {max_num} --save_tmp {output_id} --pad {pad} --gpuid {gpuid} --epicker_path {epicker_path}'
    print(f'\033[0;35m{cmd}\033[0m')
    s = os.system(cmd)
    if s != 0:
        print(f"EPicker batch failed, exit code: {s}")
    else:
        print("EPicker batch finished. Turn on 'save/use tmp file' in MPicker GUI to use the result.")
    Mpicker_epicker_batch.remove_tmpdir(tmp_dir)


def main(args):
    ero_iter = args.ero_iter
    nbin = args.nbin
    sigma = args.sigma
    rbound = args.rbound
    rbound_compete = args.rbound_compete
    knn = args.knn
    angle_thres = args.angle_thres
    region_angle_thres = args.region_angle_thres
    thres_save = args.thres_save
    thres_check = args.thres_check
    tan_near = args.tan_near
    tan_far = args.tan_far
    max_region = args.max_region
    plot_before_split = args.plot_before_split
    plot_after_split = args.plot_after_split
    plot_start = args.plot_start
    plot_final = args.plot_final
    fmask = args.fmask
    fout = args.fout
    fraw = args.fraw
    fconfig = args.fconfig
    id_start = args.id_start
    config_flatten = args.config_flatten
    epicker_model = args.epicker_model
    max_num = args.max_num
    output_id = args.output_id
    epicker_pad = args.epicker_pad
    epicker_path = args.epicker_path
    gpuid = args.gpuid

    if fout is not None:
        print("Creating MPicker folder...")
        fconfig = create_mpicker(fout, fraw, fmask)
        print(f"MPicker config created at: {fconfig}")

    if config_flatten is not None:
        flatten_parser = configparser.ConfigParser()
        flatten_parser.read(config_flatten, encoding='utf-8')
        if flatten_parser['Parameter'].getint('cylinderorder') == 0:
            print("Warning: cylinderorder=0 (simple plane fitting) is not recommended for flattening long surface, 1 or 4 is better.")
    else:
        flatten_parser = None

    # A test case for boundary file
    # fname = "AT/at645/my_boundary_6.mrc"
    # fname = "AT/my_boundary_6.mrc"
    # ero_iter = 0
    # sigma = 2.0
    # knn = 10
    # rbound = 2.5
    # rbound_compete = 2.2
    # angle_thres = 5.0
    # region_angle_thres = 90.0
    # thres_save = 2000
    # thres_check = 10000
    # tan_near = 2.0
    # tan_far = 4.0
    # max_region = 100
    # nbin = 2
    # plot_before_split = 5
    # plot_after_split = 10
    # plot_start = True
    # plot_final = True

    t0 = time()
    print("Loading data...")
    point_xyz, data_shape = load_mrc2point(fmask, ero_iter)
    point_xyz = point_xyz.astype(np.float64) # same dtype as pcd.points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_xyz)
    if nbin > 1:
        print("Downsampling...")
        pcd = pcd.voxel_down_sample(voxel_size=nbin)
        point_xyz = np.asarray(pcd.points)
    if sigma > 0:
        print("Smoothing...")
        point_xyz = smooth_point(point_xyz, sigma=sigma, k=knn)
        pcd.points = o3d.utility.Vector3dVector(point_xyz)
    print(f"{time()-t0:.1f}s.")

    if plot_start:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd], width=700, height=700)

    regions, regions_normal = region_grow_3d(pcd, knn=knn, rbound=rbound, angle_thres=angle_thres, region_angle_thres=region_angle_thres)

    lengths = np.array([len(r) for r in regions])
    mask = lengths > thres_save
    s = np.argsort(lengths[mask])[::-1]
    mask = np.arange(len(lengths))[mask][s]
    lengths = lengths[mask]
    regions = [regions[i] for i in mask]
    regions_normal = np.array([regions_normal[i] for i in mask], dtype=np.float32)
    print(f"Regions>{thres_save}: {len(regions)}. Points left: {lengths.sum()}/{len(point_xyz)}." )

    plot_regions(point_xyz, regions, plot_before_split)

    total = (lengths>thres_check).sum()
    tmp_str = "Cython" if use_cython else "Python"
    print(f"Further splitting {total} big regions ({tmp_str})...")
    regions_final = []
    i_big = 0
    for l, r, norm in zip(lengths, regions, regions_normal):
        if l <= thres_check:
            regions_final.append(r)
            continue
        t0 = time()
        r = np.array(r)
        points = point_xyz[r]
        zdiff_data, dist2d_data, indptr, indices = prepare_norm2z(points, norm, rbound)
        far_data = csr_matrix((zdiff_data / dist2d_data, indices, indptr)) > tan_far
        sorted_idx = np.argsort(far_data.sum(axis=1).A1) # points with less far neighbors first
        if use_cython:
            zdiff_data = zdiff_data.astype(np.float32)
            dist2d_data = dist2d_data.astype(np.float32)
            sorted_idx = sorted_idx.astype(np.int32)
            sub_regions = region_grow_2d_cython(zdiff_data, dist2d_data, indptr, indices, 
                                                sorted_idx, tan_near, tan_far, rbound_compete)
        else:
            sub_regions = region_grow_2d_python(zdiff_data, dist2d_data, indptr, indices, 
                                                sorted_idx, tan_near, tan_far, rbound_compete)
        sub_regions = [sr for sr in sub_regions if len(sr) > thres_save]
        sub_regions.sort(key=len, reverse=True)
        regions_final += [r[sr].tolist() for sr in sub_regions]
        i_big += 1
        print(f"{i_big}/{total}: {len(r)} points, split into {len(sub_regions)} regions. {time()-t0:.1f}s.")
        if i_big <= plot_after_split:
            plot_regions_all(points, sub_regions)

    regions_final.sort(key=len, reverse=True)
    if len(regions_final) > max_region:
        print(f"Limiting to top {max_region} regions.")
        regions_final = regions_final[:max_region]
    num_final = int(sum([len(r) for r in regions_final]))
    print(f"Final regions: {len(regions_final)}. Points left: {num_final}/{len(point_xyz)}.")

    if plot_final:
        plot_regions_all(point_xyz, regions_final)

    if fconfig is not None:
        fconfig = os.path.abspath(fconfig)
        surf_configs = []
        for i in tqdm(range(len(regions_final)), desc="Saving to mpicker"):
            points = point_xyz[regions_final[i]]
            f = save2mpicker(points, data_shape, i+id_start, fconfig, expand=(nbin>=2), flatten_parser=flatten_parser)
            surf_configs.append(f)
        if config_flatten is not None:
            print("Running flattening...")
            tomo_list = run_flatten(fconfig, surf_configs)
        print(f"Check results by: Mpicker_gui.py --config {fconfig}")

    if epicker_model is not None:
        run_epicker_batch(tomo_list, epicker_model, max_num, output_id, epicker_pad, epicker_path, gpuid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Auto-extract surfaces from membrane mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    io_group = parser.add_argument_group('Input / Output')
    io_group.add_argument('--fmask', type=str, required=True, help='Input mask mrc file path. If it is a boundary, set ero_iter=0 and nbin<=2.')
    io_group.add_argument('--fconfig', type=str, help='MPicker config file path, if provided, will add surfaces into it.')
    io_group.add_argument('--id_start', type=int, default=1, help="Starting surface ID number when add into MPicker.")
    io_group.add_argument('--fraw', type=str, help='Raw tomogram file path, if provided, will create new MPicker folder under fout.')
    io_group.add_argument('--fout', type=str, help='Output folder of MPicker, required if fraw is provided.')
    io_group.add_argument('--thres_save', type=int, default=3000, help='Minimum points number (after bin) to save as a surface.')
    io_group.add_argument('--thres_check', type=int, default=10000, help='Minimum points number (after bin) to further split.')
    io_group.add_argument('--max_region', type=int, default=100, help='Maximum number of surfaces to save.')
    io_group.add_argument('--config_flatten', type=str, help='If provided, will call MPicker to flatten each surface after separation, using [Parameter] of this config file.')
    epicker_group = parser.add_argument_group('EPicker Options. Can call Mpicker_epicker_batch.py to pick flattened tomos after flattening.')
    epicker_group.add_argument('--epicker_model', type=str, 
                                help='If provided, will pick flattened tomo using this EPicker model file. Open "save/use tmp file" in GUI to use the result.')
    epicker_group.add_argument('--max_num', type=int, default=500, help='Should be the same as in GUI')
    epicker_group.add_argument('--output_id', type=str, default='0', help='Should be the same as in GUI')
    epicker_group.add_argument('--epicker_pad', type=int, default=0, help='Should match the model (pad to square with this size). If 0, will detect max size automatically.')
    epicker_group.add_argument('--epicker_path', type=str, default="epicker.sh", help="As in Mpicker_epicker_batch.py")
    epicker_group.add_argument('--gpuid', type=int, default=0, help="As in Mpicker_epicker_batch.py")
    basic_group = parser.add_argument_group('Basic Parameters (3D region growing)')
    basic_group.add_argument('--ero_iter', type=int, default=1, help='Erosion iterations on input mask, 0 to skip. Increase to make membrane thinner.')
    basic_group.add_argument('--knn', type=int, default=15, help='KNN for neighbor search and normal estimation. For thin mask, can change to 10.')
    basic_group.add_argument('--nbin', type=float, default=2.5, help='Downsample size, <=1 to skip, the unit is voxel.')
    basic_group.add_argument('--sigma', type=float, default=2.5, help='Smoothing sigma, 0 to skip. Same as nbin is OK.')
    basic_group.add_argument('--angle_thres', type=float, default=5.0, 
                                help='Angle threshold (degree) between nearby normal vectors when region growing. If membranes are not well separated, try adjusting it.')
    basic_group.add_argument('--region_angle_thres', type=float, default=80.0, help='Angle threshold (degree) for region normal similarity when region growing.')
    advanced_group = parser.add_argument_group('Advanced Parameters (2D region growing, to further split large surfaces)')
    advanced_group.add_argument('--rbound', type=float, default=3.0, 
                                help='Max distance for neighbor search (also used in 3D region growing), should >= nbin. Can decrease it when decreasing nbin.')
    advanced_group.add_argument('--rbound_compete', type=float, default=2.6, 
                                help='Only for 2D region growing, should <= rbounnd and >= nbin. Can decrease it when decreasing nbin.')
    advanced_group.add_argument('--tan_near', type=float, default=2.0, help='Tangent threshold for near points in 2D region growing.')
    advanced_group.add_argument('--tan_far', type=float, default=4.0, help='Tangent threshold for far points in 2D region growing.')
    plot_group = parser.add_argument_group('Plotting Options (Requires OpenGL>=3 for Open3D visualization; Press ESC to close the window)')
    plot_group.add_argument('--plot_before_split', type=int, default=0, help='Number of surfaces to plot before splitting, 0 to skip.')
    plot_group.add_argument('--plot_after_split', type=int, default=0, help='Number of surfaces to plot during splitting, 0 to skip.')
    plot_group.add_argument('--plot_start', action='store_true', help='Plot point cloud after preprocessing.')
    plot_group.add_argument('--plot_final', action='store_true', help='Plot final surfaces after auto extract.')

    args = parser.parse_args()

    # print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))

    if args.fconfig is None and args.fout is None and args.fraw is None:
        print("Will not output any file.")
    if args.fconfig is not None and args.fout is not None and args.fraw is not None:
        raise ValueError("Cannot provide both fconfig and (fout, fraw).")
    if args.fout is not None and args.fraw is None:
        raise ValueError("Please provide fraw when providing fout.")
    if args.fout is None and args.fraw is not None:
        raise ValueError("Please provide fout when providing fraw.")
    if args.fconfig is not None:
        assert args.id_start >= 1, "id_start should be >= 1."
        assert os.path.isfile(args.fconfig), f"Config file {args.fconfig} does not exist!"
    if args.fout is not None:
        assert args.id_start >= 1, "id_start should be >= 1."
        os.makedirs(args.fout, exist_ok=True)
    if args.config_flatten is not None:
        assert os.path.isfile(args.config_flatten), f"Flatten config file {args.config_flatten} does not exist!"
    if args.epicker_model is not None:
        assert args.config_flatten is not None, "Picking using EPicker requires flattening first, please provide config_flatten."
        assert os.path.isfile(args.epicker_model), f"EPicker model file {args.epicker_model} does not exist!"
        assert " " not in args.output_id, "output_id should not contain space."
    if not (args.rbound_compete >= args.nbin and args.rbound >= args.rbound_compete):
        print("rbound >= rbound_compete >= nbin is not satisfied, not recommended.")
    if args.rbound > 2 * args.nbin:
        print("rbound > 2 * nbin, may too large.")
    if args.region_angle_thres > 90.0:
        print("region_angle_thres should <=90.0, set to 90.")
        args.region_angle_thres = 90.0

    main(args)