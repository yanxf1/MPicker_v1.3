# cython: boundscheck=False, wraparound=False

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
cimport numpy as np
from libc.math cimport fabs, sqrt

ctypedef np.float32_t FLOAT32
ctypedef np.int32_t   INT32
ctypedef np.uint8_t   UINT8


cdef class FastDeque:
    cdef:
        INT32[:] buffer
        int capacity
        int head
        int tail
        int size

    def __init__(self, int capacity):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=np.int32)
        self.head = 0
        self.tail = 0
        self.size = 0

    cpdef void append(self, int x):
        cdef int newcap, i, idx
        cdef np.ndarray[INT32, ndim=1] newarr
        cdef INT32[:] newbuf
        if self.size >= self.capacity:
            # resize to double capacity and preserve order
            newcap = self.capacity * 2
            newarr = np.empty(newcap, dtype=np.int32)
            newbuf = newarr
            for i in range(self.size):
                idx = (self.head + i) % self.capacity
                newbuf[i] = self.buffer[idx]
            self.buffer = newbuf
            self.capacity = newcap
            self.head = 0
            self.tail = self.size
        self.buffer[self.tail] = x
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    cpdef int popleft(self):
        if self.size == 0:
            raise IndexError("popleft from empty FastDeque")
        cdef int val = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return val

    cpdef bint empty(self):
        return self.size == 0

    cpdef void clear(self):
        self.size = 0
        self.head = 0
        self.tail = 0


def region_grow_cython(np.ndarray[FLOAT32, ndim=2] normals not None,
                       np.ndarray[INT32, ndim=1] indptr not None,
                       np.ndarray[INT32, ndim=1] indices not None,
                       np.ndarray[FLOAT32, ndim=1] cosines_all not None,
                       np.ndarray[INT32, ndim=1] sorted_indices not None,
                       float cos_thres,
                       float region_cos_thres,
                       np.ndarray[UINT8, ndim=1] visited=None):

    cdef int N = normals.shape[0]
    cdef np.float32_t[:, :] nrm = normals
    cdef np.int32_t[:] iptr = indptr
    cdef np.int32_t[:] idxs = indices
    cdef np.float32_t[:] cos_all = cosines_all
    cdef np.int32_t[:] sidx = sorted_indices
    if visited is None:
        visited = np.zeros(N, dtype=np.uint8)
    cdef np.uint8_t[:] visited_view = visited

    regions = []
    regions_normal = []
    cdef int ii, start, end, cur, e, nbr, idx0
    cdef float rx, ry, rz, norm_r, dotv, nx, ny, nz, cnx, cny, cnz
    cdef FastDeque queue = FastDeque(N)

    for ii in range(sidx.shape[0]):
        idx0 = sidx[ii]
        if visited_view[idx0]:
            continue
        queue.clear()
        queue.append(idx0)
        region_list = []
        # region normal sum
        rx = ry = rz = 0.0
        while not queue.empty():
            cur = queue.popleft()
            if visited_view[cur]:
                continue
            visited_view[cur] = 1
            region_list.append(cur)
            # update region normal sum (use sign same as existing logic)
            nx = nrm[cur, 0]; ny = nrm[cur, 1]; nz = nrm[cur, 2]
            if (rx * nx + ry * ny + rz * nz) >= 0.0:
                rx += nx; ry += ny; rz += nz
            else:
                rx -= nx; ry -= ny; rz -= nz
            norm_r = sqrt(rx*rx + ry*ry + rz*rz)
            if norm_r < 1e-12:
                cnx = nx; cny = ny; cnz = nz
            else:
                cnx = rx / norm_r; cny = ry / norm_r; cnz = rz / norm_r

            start = iptr[cur]
            end = iptr[cur+1]
            for e in range(start, end):
                nbr = idxs[e]
                if visited_view[nbr]:
                    continue
                # edge cosine threshold
                if cos_all[e] < cos_thres:
                    continue
                # region cosine check: dot(region_normal, normals[nbr])
                dotv = cnx * nrm[nbr,0] + cny * nrm[nbr,1] + cnz * nrm[nbr,2]
                if fabs(dotv) < region_cos_thres:
                    continue
                queue.append(nbr)

        regions.append(np.asarray(region_list, dtype=np.int32))
        regions_normal.append([cnx, cny, cnz])
    return regions, regions_normal


def region_grow_2d_cython(np.ndarray[FLOAT32, ndim=1] zdiff_data not None,
                          np.ndarray[FLOAT32, ndim=1] dist2d_data not None, 
                          np.ndarray[INT32, ndim=1] indptr not None,
                          np.ndarray[INT32, ndim=1] indices not None,
                          np.ndarray[INT32, ndim=1] sorted_indices not None,
                          float tan_near,
                          float tan_far,
                          float dist_compete):

    cdef np.float32_t[:] zd = zdiff_data
    cdef np.float32_t[:] dd = dist2d_data
    cdef np.int32_t[:] iptr = indptr
    cdef np.int32_t[:] idxs = indices
    cdef np.int32_t[:] sidx = sorted_indices

    cdef int N = indptr.shape[0] - 1
    cdef int i, ii, start, end, cur, e, nbr
    cdef float tan_val
    regions = []
    # visited: global visited across seeds
    visited = np.zeros(N, dtype=np.uint8)
    cdef np.uint8_t[:] visited_view = visited

    # declare memoryview variables up-front (cannot cdef-initialize later)
    cdef np.uint8_t[:] visited_this_view = np.empty(N, dtype=np.uint8)
    cdef np.uint8_t[:] far_points_view = np.empty(N, dtype=np.uint8)

    cdef FastDeque queue_idx = FastDeque(N)
    cdef FastDeque queue_state = FastDeque(N)

    # temporary arrays allocated once per seed
    for i in range(sidx.shape[0]):
        ii = sidx[i]
        if visited_view[ii]:
            continue
        # visited_this starts as copy of visited
        visited_this_view[:] = visited_view[:]
        far_points_view[:] = 0

        region_list = []
        # store indices and state (1 for is_this True, 0 for False)
        queue_idx.clear()
        queue_idx.append(ii)
        queue_state.clear()
        queue_state.append(1)

        while not queue_idx.empty():
            cur = queue_idx.popleft()
            state = queue_state.popleft()
            if visited_this_view[cur]:
                continue

            if state == 1:
                # 'this' branch
                if far_points_view[cur]:
                    continue
                visited_this_view[cur] = 1
                region_list.append(cur)
                visited_view[cur] = 1
                start = iptr[cur]
                end = iptr[cur+1]
                # iterate neighbors
                for e in range(start, end):
                    nbr = idxs[e]
                    if visited_this_view[nbr]:
                        continue
                    tan_val = zd[e] / dd[e]
                    if tan_val <= tan_near:
                        # near edge -> push as 'this' state
                        queue_idx.append(nbr)
                        queue_state.append(1)
                for e in range(start, end):
                    nbr = idxs[e]
                    if visited_this_view[nbr]:
                        continue
                    tan_val = zd[e] / dd[e]
                    if tan_val > tan_far:
                        # far edge -> mark far_points and push as other state
                        far_points_view[nbr] = 1
                        queue_idx.append(nbr)
                        queue_state.append(0)
            else:
                # 'other' branch (is_this == False)
                visited_this_view[cur] = 1
                start = iptr[cur]
                end = iptr[cur+1]
                for e in range(start, end):
                    nbr = idxs[e]
                    if visited_this_view[nbr]:
                        continue
                    # only traverse near edges from other-state nodes
                    tan_val = zd[e] / dd[e]
                    if tan_val <= tan_near and dd[e] <= dist_compete:
                        far_points_view[nbr] = 1
                        queue_idx.append(nbr)
                        queue_state.append(0)

        # mark global visited for all nodes in this region
        if region_list:
            regions.append(region_list)

    return regions