#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
from os import path

from data_structs import Dataset, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def pretrain(restore_from=None):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc")

    # Create a Dataset from a SMILES file
    if path.isfile('./data/vecs.dat'):
        data = Dataset(voc, "data/mols.smi", vec_file='./data/vecs.dat')
    else:
        data = Dataset(voc, "data/mols.smi", vec_file=None)

    batch_size = 100

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                        collate_fn=Dataset.collate_fn)
    Prior = RNN(voc, len(data[0][1]))

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))
    
    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)
    for epoch in range(1, 6):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        for step, (smi_batch, vec_batch) in tqdm(enumerate(loader), total=len(loader)):

            # Sample from DataLoader
            seqs = smi_batch.long()
            vecs = vec_batch.float()

            #Could calculate a loss here that is comparing vector in and out?

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs, vecs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data.item()))
                seqs, likelihood, _ = Prior.sample(batch_size, vecs)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

if __name__ == "__main__":
    pretrain()
