import argparse
import os
import sys
os.chdir('../')
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm import tqdm
from glob import glob
from collections import Counter
from pathlib import Path

from src.evaluation import eval_atom_type, scoring_func, misc, eval_bond_length
from src.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    r = []
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        r.append(n_mol / len(all_ring_sizes))
    for i in range(0, 7):
        logger.info(f'ring size: {ring_size+3} ratio: {r[i] / sum(r):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='outputs')
    parser.add_argument('--sample_path', type=str, default='samples/test/last_T=5000')
    parser.add_argument('--dataset', type=str, choices=['Crossdocked', 'PDBbind'])
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'], default='vina_dock')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    args = parser.parse_args()

    result_path = os.path.join(args.output_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    in_dir = Path(args.sample_path)
    all_dirs = [path for path in in_dir.glob("[!.]*") if os.path.isdir(path)]
    num_samples = len(all_dirs)
    logger.info(f'Load generated data done! {num_samples} examples in total.')
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist, all_results = [], [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    example_idx = -1
    for subdir in tqdm(all_dirs):
        example_idx += 1
        print(example_idx)
        valid_files = [
            int(fname.split('_')[0])
            for fname in os.listdir(subdir)
            if fname.endswith('_ligand.sdf') and not fname.startswith('.')
        ]
        in_prot = str(subdir).replace(args.sample_path, 'processed/test' if args.dataset == 'Crossdocked' else 'processedP/test')
        index = in_prot.find('pocket')
        in_mol = in_prot + '.sdf'
        in_prot = in_prot[:index] + 'pocket10.pdb'
        upper_bound = max(valid_files) + 1
        success = []
        for i in range(upper_bound):
            in_mol = str(Path(subdir, f'{i}_ligand.sdf'))
            mol = Chem.SDMolSupplier(in_mol)
            mol = next(mol)
            if mol == None: continue
            pred_pos = mol.GetConformer(0).GetPositions()
            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                if args.docking_mode == 'qvina':
                    vina_task = VinaDockingTask(in_prot, mol)
                    vina_results = vina_task.run_sync()
                elif args.docking_mode in ['vina_score', 'vina_dock']:
                    vina_task = VinaDockingTask(in_prot, mol)
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if args.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None
            except:
                continue

            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            results.append({
                'mol': mol,
                'chem_results': chem_results,
                'vina': vina_results,
                'ligand': in_mol
            })
            success.append({
                'mol': mol,
                'chem_results': chem_results,
                'vina': vina_results,
                'id': example_idx
            })
        all_results.append(success)
    torch.save(all_results, os.path.join(result_path, 'metrics.pt'))
    logger.info(f'Evaluate done! {num_samples} samples in total.')

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
    mols = np.argsort(vina_score_only)[:10000]
    qed = np.array([r['chem_results']['qed'] for r in results])[mols]
    sa = np.array([r['chem_results']['sa'] for r in results])[mols]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    try:
        logp = np.array([r['chem_results']['logp'] for r in results])[mols]
        lipinski = np.array([r['chem_results']['lipinski'] for r in results])[mols]
        logger.info('logp:   Mean: %.3f Median: %.3f' % (np.mean(logp), np.median(logp)))
        logger.info('lipinski:    Mean: %.3f Median: %.3f' % (np.mean(lipinski), np.median(lipinski)))
    except:
        print('')

    if args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = np.array([r['vina']['score_only'][0]['affinity'] for r in results])[mols]
        vina_min = np.array([r['vina']['minimize'][0]['affinity'] for r in results])[mols]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = np.array([r['vina']['dock'][0]['affinity'] for r in results])[mols]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)
