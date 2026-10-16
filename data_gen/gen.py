"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import pymesh
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from subprocess import Popen, PIPE


# parse MSMS output
def read_msms(data_dir, prefix):
    mesh_prefix = os.path.join(data_dir, prefix)
    assert os.path.isfile(mesh_prefix + '.vert')
    assert os.path.isfile(mesh_prefix + '.face')

    # vertices
    with open(mesh_prefix + '.vert') as f:
        vert_data = f.read().rstrip().split('\n')
    num_verts = int(vert_data[2].split()[0])
    assert num_verts == len(vert_data) - 3
    vertices = []
    for idx in range(3, len(vert_data)):
        ifields = vert_data[idx].split()
        assert len(ifields) == 10
        vertices.append(ifields[:3])
    assert len(vertices) == num_verts

    # faces
    with open(mesh_prefix + '.face') as f:
        face_data = f.read().rstrip().split('\n')
    num_faces = int(face_data[2].split()[0])
    assert num_faces == len(face_data) - 3
    faces = []
    for idx in range(3, len(face_data)):
        ifields = face_data[idx].split()
        assert len(ifields) == 5
        faces.append(ifields[:3])  # one-based, to be converted
    assert len(faces) == num_faces

    # solvent excluded surface info
    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int) - 1  # convert to zero-based indexing
    assert np.amin(faces) == 0

    return vertices, faces


# skip surface with poor mesh quality
def check_mesh_validity(mesh, check_triangles=False):
    mesh.enable_connectivity()
    verts, faces = mesh.vertices, mesh.faces

    # check if a manifold is all-connected using BFS
    visited = np.zeros(len(verts)).astype(bool)
    groups = []
    for ivert in range(len(verts)):
        if visited[ivert]:
            continue
        old_visited = visited.copy()
        queue = [ivert]
        visited[ivert] = True
        while queue:
            curr = queue.pop(0)
            for nbr in mesh.get_vertex_adjacent_vertices(curr):
                if not visited[nbr]:
                    queue.append(nbr)
                    visited[nbr] = True
        groups.append(np.where(np.logical_xor(old_visited, visited))[0])
    groups = sorted(groups, key=lambda x:len(x), reverse=True)
    assert sum(len(ig) for ig in groups) == sum(visited) == len(verts)

    # check for isolated vertices
    valid_verts = np.unique(faces)
    has_isolated_verts = verts.shape[0] != len(valid_verts)

    # check for faces with duplicate vertices
    df = pd.DataFrame(faces)
    df = df[df.nunique(axis=1) == 3]
    has_duplicate_verts = df.shape[0] != mesh.num_faces

    # check for abnormal triangles
    if check_triangles:
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        e1 = v3 - v2
        e2 = v1 - v3
        e3 = v2 - v1
        L1 = np.linalg.norm(e1, axis=1)
        L2 = np.linalg.norm(e2, axis=1)
        L3 = np.linalg.norm(e3, axis=1)
        cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
        cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
        cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)
        cos123 = np.concatenate((cos1.reshape(-1, 1),
                                 cos2.reshape(-1, 1),
                                 cos3.reshape(-1, 1)), axis=-1)
        valid_faces = np.where(np.all(1 - cos123**2 >= 1E-5, axis=-1))[0]
        has_abnormal_triangles = faces.shape[0] != len(valid_faces)
    else:
        has_abnormal_triangles = False

    return groups, has_isolated_verts, has_duplicate_verts, has_abnormal_triangles


def refine_mesh(mesh_msms, resolution=1.0):
    mesh, _ = pymesh.remove_duplicated_vertices(mesh_msms, 1E-6)
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, _ = pymesh.split_long_edges(mesh, resolution)
    return mesh.vertices

# use MSMS to compute molecular solvent excluded surface
def compute_msms(gdb_id, data_root='data_gen/mesh_train', msms_bin='MSMS/msms.x86_64Linux2.2.6.1'):
    # specify IO dir
    data_dir = os.path.join(data_root, gdb_id)
    # xyzrn file
    xyzrn_file = os.path.join(data_dir, f'{gdb_id}.xyzrn')
    assert os.path.isfile(xyzrn_file)
    mesh_prefix = os.path.join(data_dir, f'{gdb_id}')

    # run MSMS
    msms_args = [msms_bin, '-if', xyzrn_file, '-of', mesh_prefix, \
                    '-probe_radius', '1.0', '-density', '3.0']
    proc = Popen(msms_args, stdout=PIPE, stderr=PIPE)
    _, stderr = proc.communicate()
    # skip if MSMS failed
    errmsg = stderr.decode('utf-8')
    if 'ERROR' in errmsg or 'WARNING' in errmsg:
        print(f'skip {gdb_id}\n  {errmsg}')
        return []
    if not (os.path.isfile(mesh_prefix+'.vert') and \
            os.path.isfile(mesh_prefix+'.face')):
        print(f'skip {gdb_id} due to missing MSMS output', flush=True)
        return []

    # check mesh validity
    verts, faces = read_msms(data_dir, gdb_id)
    mesh = pymesh.form_mesh(verts, faces)
    m = refine_mesh(mesh)
    return m
