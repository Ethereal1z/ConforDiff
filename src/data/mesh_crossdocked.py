import os
import torch
import numpy as np
from tqdm.auto import tqdm
from data_gen.data import parse_sdf_file, convert_to_xyzrn
from data_gen.gen import compute_msms


data_split = torch.load('data/crossdocked_data/split_by_name.pt')
data_prefix = 'data/crossdocked_data/crossdocked_pocket10'
manifold = []
for i, (pocket_fn, ligand_fn) in enumerate(tqdm(data_split['train'])):
    if pocket_fn is None: continue
    try:
        element, pos = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
        id = convert_to_xyzrn(i, element, pos, 'data_gen/mesh_train')
        m = compute_msms(id, 'data_gen/mesh_train')
    except:
        m = np.array([])
    manifold.append(torch.from_numpy(m))
torch.save(manifold, 'data/mtrain.pt')

manifold = []
for i, (pocket_fn, ligand_fn) in enumerate(tqdm(data_split['test'])):
    try:
        element, pos = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
        id = convert_to_xyzrn(i, element, pos, 'data_gen/mesh_test')
        m = compute_msms(id, 'data_gen/mesh_test')
    except:
        m = np.array([])
    manifold.append(torch.from_numpy(m))
torch.save(manifold, 'data/mtest.pt')

