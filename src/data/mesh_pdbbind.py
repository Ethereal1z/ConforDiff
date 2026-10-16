import os
import numpy as np
import torch
import pickle
from tqdm.auto import tqdm
from data_gen.data import parse_sdf_file, convert_to_xyzrn
from data_gen.gen import compute_msms


data_split = torch.load('data/PDBbind_pocket10_split.pt')
with open('data/PDBbind_v2020/index.pkl', "rb") as f:
    index = pickle.load(f)
manifold = []
dataroot = 'data_gen/mesh_testp'
data_prefix = 'data/PDBbind_v2020'
for i, iid in enumerate(tqdm(data_split['test'])):
    pocket_fn = index[iid][0][:-len('rotein.pdb')] + "ocket.pdb"
    ligand_fn = index[iid][1]
    try:
        element, pos = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
        id = convert_to_xyzrn(i, element, pos, out_root=dataroot)
        m = compute_msms(id, dataroot)
    except:
        m = np.array([])
    manifold.append(torch.from_numpy(m))
torch.save(manifold, 'data/mtestp.pt')