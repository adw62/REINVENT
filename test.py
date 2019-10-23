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
from utils import seq_to_smiles

def black_box(load_weights='data/Prior.ckpt', batch_size=1):

    voc = Vocabulary(init_from_file="data/Voc")
    Prior = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(load_weights))
    else:
        Prior.rnn.load_state_dict(torch.load(load_weights, map_location=lambda storage, loc: storage))

    #data = []
    #for param in Prior.rnn.gru_1.parameters():
    #    data.append(param)

    data = Prior.rnn.gru_1.weight_ih
    data = data*0.0
    print(data)
    Prior.rnn.gru_1.weight_ih = torch.nn.Parameter(data)

    seqs, prior_likelihood, entropy = Prior.sample(batch_size)
    smiles = seq_to_smiles(seqs, voc)
    print(smiles)

black_box()