import os
import shutil
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
vdw_radii_dict = np.array([0, 1.1, 0, 0, 0, 0, 1.7, 1.55, 1.52, 1.47, 0, 0, 0, 0, 0, 1.8, 1.8, 1.75,
                          0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,1.9,1.95,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,2.18], dtype=np.float32)

def parse_sdf_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
    element = np.array(element, dtype=np.int)

    return element, pos


# prepare input for MSMS surface computation
def convert_to_xyzrn(id, ligand_element, ligand_pos, out_root='data_gen/mesh_train'):
    gdb_id = 'gdb_' + str(id).rjust(6, '0')
    natom = len(ligand_element)
    atom_xyz = ligand_pos
    atom_Z = ligand_element
    atom_r = vdw_radii_dict[atom_Z]
    atom_rn = [f'{atom_r[i]} 1 {atom_Z[i]}' for i in range(natom)]

    # write xyzrn file
    out_dir = os.path.join(out_root, gdb_id)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=False)
    xyzrn_path = os.path.join(out_dir, f'{gdb_id}.xyzrn')
    with open(xyzrn_path, 'w') as f:
        for i in range(natom):
            coords = '{:.6f} {:.6f} {:.6f} '.format(*atom_xyz[i])
            f.write(coords + atom_rn[i] + '\n')
    return gdb_id