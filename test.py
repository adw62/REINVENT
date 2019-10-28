#!/usr/bin/env python

import torch
import pickle
import numpy as np
import time
import os
from shutil import copyfile
import math

from model import RNN
from data_structs import Vocabulary
from torch.utils.data import DataLoader
from data_structs import VecData, MolData
from utils import seq_to_smiles

def black_box(load_weights='data/Prior.ckpt', batch_size=128):
    
    vecdata = VecData("data/test_dat/vecs_test.dat")
    data_vec = DataLoader(vecdata, batch_size=batch_size, shuffle=True, drop_last=True,
                          collate_fn=MolData.collate_fn)

    voc = Vocabulary(init_from_file="data/Voc")
    Prior = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(load_weights))
    else:
        Prior.rnn.load_state_dict(torch.load(load_weights, map_location=lambda storage, loc: storage))

    for vec_batch in data_vec:
        seqs, prior_likelihood, entropy = Prior.sample(batch_size, vec_batch)
        smiles = seq_to_smiles(seqs, voc)
        print(smiles)

black_box()
