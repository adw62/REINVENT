#!/usr/bin/env python

import torch
import pickle
import numpy as np
import time
from os import path
from shutil import copyfile
import math
from rdkit import Chem
from rdkit import rdBase

from model import RNN
from data_structs import Vocabulary
from torch.utils.data import DataLoader
from data_structs import Dataset
from utils import seq_to_smiles, get_latent_vector

def black_box(load_weights='data/Prior.ckpt', batch_size=1):

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc")

    # Create a Dataset from a SMILES file
    if path.isfile('./data/vecs.dat'):
        data = Dataset(voc, "data/mols.smi", vec_file='./data/vecs.dat')
    else:
        data = Dataset(voc, "data/mols.smi", vec_file=None)

    Prior = RNN(voc, len(data[0][1]))

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(load_weights))
    else:
        Prior.rnn.load_state_dict(torch.load(load_weights, map_location=lambda storage, loc: storage))
    
    data = [data[0][1]]
    for test_vec in data:
        print('Test vector {}'.format(test_vec))
        test_vec = test_vec.float()
        valid = 0
        num_smi = 100
        all_smi = []
        for i in range(num_smi):
            seqs, prior_likelihood, entropy = Prior.sample(batch_size, test_vec)
            smiles = seq_to_smiles(seqs, voc)[0]
            all_smi.append(smiles)
            if Chem.MolFromSmiles(smiles):
                        valid += 1

        for smi in all_smi:
            print(smi)
        print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(range(num_smi))))

black_box()
