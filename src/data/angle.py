import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem


class PocketSelect(Select):
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues


def get_ligand_coords(ligand_file):
    mol = Chem.SDMolSupplier(ligand_file)[0]
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])

    return np.array(coords)


def extract_pocket(protein_file, ligand_file, output_file, cutoff=10.0):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_file)

    ligand_coords = get_ligand_coords(ligand_file)

    pocket_residues = set()

    for residue in structure.get_residues():

        for atom in residue.get_atoms():

            coord = atom.coord
            dists = np.linalg.norm(ligand_coords - coord, axis=1)

            if np.min(dists) <= cutoff:
                pocket_residues.add(residue)
                break

    io = PDBIO()
    io.set_structure(structure)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    io.save(output_file, PocketSelect(pocket_residues))



if __name__ == "__main__":
    # Read data split
    data_split = torch.load('/home/u1/下载/PDBbind_pocket10_split.pt')
    with open('/home/u1/下载/PDBbind_v2020/index.pkl', "rb") as f:
        index = pickle.load(f)

    for split in data_split.keys():
        split = 'train'
        print(f"Processing {split} dataset...")

        pbar = tqdm(data_split[split])
        for id in pbar:
            pocket_fn = index[id][0]  # failed: 18989
            ligand_fn = index[id][1]

            sdffile = '/home/u1/下载/PDBbind_v2020/' + f'{ligand_fn}'
            pdbfile = '/home/u1/下载/PDBbind_v2020/' + f'{pocket_fn}'
            outfile = '/home/u1/下载/PDBbind_train10A/' + f'{pocket_fn}'
            try:
                extract_pocket(pdbfile, sdffile, outfile, 10.0)
            except:
                print('failed:', id)
        break