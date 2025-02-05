#!/usr/bin/env python

import numpy as np 
import os  
from rdkit import Chem
from glob import glob
import parmed
from deepdih.mollib.fragment import Fragmentation
import sys
from tblite.ase import TBLite
#from deepdih.utils import (
#    get_mol_with_indices,
#    TorEmbeddedMolecule,
#    map_mol1_to_mol2,
#    constrained_embed,
#    SpecbondEmbeddedMolecule,
#    SpecangleEmbeddedMolecule,
#    SpecdihedralEmbeddedMolecule,
#    SpecQEmbeddedMolecule
#)
from deepdih.utils.geometry import calc_max_disp, calc_rmsd 
import deepdih 
from multiprocessing import Pool
from deepdih.utils.embedding import get_embed
import torch 

opera = 'train_process' 
if opera == 'train':
    # 这里我们需要修改下deepdih里面的embedmol, 使得torsion可以自己指定
    # 已经定义了函数 SpecdihedralEmbeddedMolecule 可用于替代embedmol 
    import pickle 
    f_names = glob('./semisucced/mol_*/train_data.npz') + glob('./succed/mol_*/train_data.npz') + glob('./tryagain/mol_*/train_data.npz')
    #f_names = glob('./mol_*/train_data.npz')
    f_names_remain = []
    training_data = []; training_data1 = []; training_data2 = []
    for f_name in f_names:
        f_dir = os.path.dirname(f_name)
        f_npz = os.path.join(f_dir, 'train_data.npz')
        f_scan = os.path.join(f_dir, 'dih_scan.sdf')
        torsions = []
        with open(os.path.join(f_dir, 'mol.itp'), 'r') as f:
            for line in f:
                if 'missing' in line:
                    line = line.strip().split()
                    torsions.append((int(line[0])-1, int(line[1])-1, int(line[2])-1, int(line[3])-1))
        data = np.load(f_npz, allow_pickle=True)
        _coords = data['coord_mmopt']; _qm_energies = data['qm_from_mmopt']
        _mm_energies = data['mm_from_mmopt']; _reasons = data['reason']
        _qm_energies_fromxtb = data['qm_from_xtbopt']
        _xtb_energies_fromxtb = data['xtb_from_xtbopt']
        coords = []; qm_energies = []; mm_energies = []; qm_energies_fromxtb = []; xtb_energies_fromxtb = [] 
        for i, reason in enumerate(_reasons):
            #if reason == True:
            if 1 == 1:
            #if reason == True and _mm_energies[i] < 0.05:
                coords.append(_coords[i])
                qm_energies.append(_qm_energies[i])
                mm_energies.append(_mm_energies[i])
                qm_energies_fromxtb.append(_qm_energies_fromxtb[i])
                xtb_energies_fromxtb.append(_xtb_energies_fromxtb[i])
        
        if len(qm_energies) > len(_coords) * 0.8:
            print(f_name)
            qm_energies = np.array(qm_energies); mm_energies = np.array(mm_energies); qm_energies_fromxtb = np.array(qm_energies_fromxtb)
            xtb_energies_fromxtb = np.array(xtb_energies_fromxtb)
            qm_energies = qm_energies - qm_energies.mean()
            mm_energies = mm_energies - mm_energies.mean()
            qm_energies_fromxtb = qm_energies_fromxtb - qm_energies_fromxtb.mean()
            xtb_energies_fromxtb = xtb_energies_fromxtb - xtb_energies_fromxtb.mean()

            delta_energies = qm_energies - mm_energies # in Hartree
            delta_energies = delta_energies / deepdih.utils.EV_TO_HARTREE * deepdih.utils.EV_TO_KJ_MOL
            delta_energies_fromxtb = qm_energies_fromxtb - mm_energies
            delta_energies_fromxtb = delta_energies_fromxtb / deepdih.utils.EV_TO_HARTREE * deepdih.utils.EV_TO_KJ_MOL
            delta_xtb_energies_fromxtb = xtb_energies_fromxtb - mm_energies
            delta_xtb_energies_fromxtb = delta_xtb_energies_fromxtb / deepdih.utils.EV_TO_HARTREE * deepdih.utils.EV_TO_KJ_MOL

            mols = deepdih.utils.read_sdf(f_scan)

            embedded_mol = deepdih.utils.embedding.SpecdihedralEmbeddedMolecule(mols[0], \
                        conf=coords, target=delta_energies, all_tors= torsions)
            embedded_mol1 = deepdih.utils.embedding.SpecdihedralEmbeddedMolecule(mols[0], \
                        conf=coords, target=delta_energies_fromxtb, all_tors= torsions)
            embedded_mol2 = deepdih.utils.embedding.SpecdihedralEmbeddedMolecule(mols[0], \
                        conf=coords, target=delta_xtb_energies_fromxtb, all_tors= torsions)
            
            training_data.append(embedded_mol)
            training_data1.append(embedded_mol1)
            training_data2.append(embedded_mol2)
            f_names_remain.append(f_name)
    with open('fname.txt','w') as f:
        for ii in f_names_remain:
            f.write(f'{ii} \n')

    with open('data_qm_frommmopt.pkl','wb') as f:
        pickle.dump(training_data, f)
    with open('data_qm_fromxtbopt.pkl','wb') as f:
        pickle.dump(training_data1, f)
    with open('data_xtb_fromxtbopt.pkl','wb') as f:
        pickle.dump(training_data2, f)
        # train the model 
        #params = deepdih.finetune.finetune_workflow(training_data, n_fold=3)
        #with open('params.pkl','wb') as f:
        #    pickle.dump(params, f)
 
elif opera == 'train_process':
    import pickle 
    #_names = ['./frag_mol/mol_514/train_data.npz']
    f_names = glob('./semisucced/mol_*/train_data.npz') + glob('./succed/mol_*/train_data.npz') + glob('./tryagain/mol_*/train_data.npz')
    mols_reasonable = []
    with open('train_file.txt','r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split('/')
            line_2 = line[2].split('_')
            mol = line_2[0] + '_' + line_2[1]
            mols_reasonable.append(mol)
    print(mols_reasonable)
    
    f_names_remain = []
    for f_name in f_names:
        f_name0 = f_name.strip().split('/')[2]
        if f_name0 in mols_reasonable:
            f_names_remain.append(f_name)
    print(f_names_remain)

    print(len(f_names_remain))
    print(len(f_names))
    print('*****end******')

    training_data = []; training_data1 = []; training_data2 = []
    for f_name in f_names_remain:
        f_dir = os.path.dirname(f_name)
        f_npz = os.path.join(f_dir, 'train_data.npz')
        f_scan = os.path.join(f_dir, 'dih_scan.sdf')
        torsions = []
        with open(os.path.join(f_dir, 'mol.itp'), 'r') as f:
            for line in f:
                if 'missing' in line:
                    line = line.strip().split()
                    torsions.append((int(line[0])-1, int(line[1])-1, int(line[2])-1, int(line[3])-1))
        data = np.load(f_npz, allow_pickle=True)
        _coords = data['coord_mmopt']; _qm_energies = data['qm_from_mmopt']
        _mm_energies = data['mm_from_mmopt']; _reasons = data['reason']
        _qm_energies_fromxtb = data['qm_from_xtbopt']
        _xtb_energies_fromxtb = data['xtb_from_xtbopt']
        coords = []; qm_energies = []; mm_energies = []; qm_energies_fromxtb = []; xtb_energies_fromxtb = [] 
        for i, reason in enumerate(_reasons):
            #if reason == True:
            if 1 == 1:
            #if reason == True and _mm_energies[i] < 0.05:
                coords.append(_coords[i])
                qm_energies.append(_qm_energies[i])
                mm_energies.append(_mm_energies[i])
                qm_energies_fromxtb.append(_qm_energies_fromxtb[i])
                xtb_energies_fromxtb.append(_xtb_energies_fromxtb[i])
        delta = 0.
        if len(qm_energies) > len(_coords) * 0.8:
            print(f_name)
            qm_energies = np.array(qm_energies); mm_energies = np.array(mm_energies); qm_energies_fromxtb = np.array(qm_energies_fromxtb)
            xtb_energies_fromxtb = np.array(xtb_energies_fromxtb)
            qm_energies = qm_energies - qm_energies.mean()
            mm_energies = mm_energies - mm_energies.mean()
            qm_energies_fromxtb = qm_energies_fromxtb - qm_energies_fromxtb.mean()
            xtb_energies_fromxtb = xtb_energies_fromxtb - xtb_energies_fromxtb.mean()

            delta_energies = qm_energies - mm_energies # in Hartree
            delta_energies = delta_energies / deepdih.utils.EV_TO_HARTREE * deepdih.utils.EV_TO_KJ_MOL
            delta_energies_fromxtb = qm_energies_fromxtb - mm_energies
            delta_energies_fromxtb = delta_energies_fromxtb / deepdih.utils.EV_TO_HARTREE * deepdih.utils.EV_TO_KJ_MOL
            delta_xtb_energies_fromxtb = xtb_energies_fromxtb - mm_energies
            delta_xtb_energies_fromxtb = delta_xtb_energies_fromxtb / deepdih.utils.EV_TO_HARTREE * deepdih.utils.EV_TO_KJ_MOL
            if np.max(delta_energies) > delta:
                delta = np.max(delta_energies)
            mols = deepdih.utils.read_sdf(f_scan)

            embedded_mol = deepdih.utils.embedding.SpecdihedralEmbeddedMolecule(mols[0], \
                        conf=coords, target=delta_energies, all_tors= torsions)
            embedded_mol1 = deepdih.utils.embedding.SpecdihedralEmbeddedMolecule(mols[0], \
                        conf=coords, target=delta_energies_fromxtb, all_tors= torsions)
            embedded_mol2 = deepdih.utils.embedding.SpecdihedralEmbeddedMolecule(mols[0], \
                        conf=coords, target=delta_xtb_energies_fromxtb, all_tors= torsions)
            
            training_data.append(embedded_mol)
            training_data1.append(embedded_mol1)
            training_data2.append(embedded_mol2)
    print('************')
    print(delta)
    with open('data_qm_frommmopt_remain.pkl','wb') as f:
        pickle.dump(training_data, f)
    with open('data_qm_fromxtbopt_remain.pkl','wb') as f:
        pickle.dump(training_data1, f)
    with open('data_xtb_fromxtbopt_remain.pkl','wb') as f:
        pickle.dump(training_data2, f)
        # train the model 
        #params = deepdih.finetune.finetune_workflow(training_data, n_fold=3)
        #with open('params.pkl','wb') as f:
        #    pickle.dump(params, f)

elif opera == 'run_train':
    import pickle 
    import sys
    f_name = sys.argv[1]
    task_name = sys.argv[2]
    #num0 = int(sys.argv[1])
    #num1 = int(sys.argv[2])
    all_files = []
    with open('fname.txt','r') as f:
        for line in f:
            all_files.append(line.strip().split('/')[-2]+f'_{task_name}')
    cwd = os.path.abspath(os.getcwd())
    #for ii in range(10):
    for ii in range(0, len(all_files)):
        if os.path.exists(f'{all_files[ii]}'):
            pass
        else:
            os.mkdir(f'{all_files[ii]}')
        #os.chdir(f'{all_files[ii]}')
        os.system(f'python train_mol.py {f_name} {ii} > res.out')
        #os.chdir(cwd)
        os.system(f'mv res.out {all_files[ii]}')
        os.system(f'mv params.pkl {all_files[ii]}')
        #params = deepdih.finetune.finetune_workflow(training_data[num0:num1], n_fold=1)
        #with open(f'params.pkl','wb') as f:
        #    pickle.dump(params, f)
