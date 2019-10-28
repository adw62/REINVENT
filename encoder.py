#!/usr/bin/env python

from rdkit import Chem
import torch

class encoder(object):
    def __init__(self):
        pass

    def process_file(self, filename):
        with open(filename) as file:
            for line in file:
                print(encoder.encode(line))

    def encode(self):
        mol = Chem.MolFromSmiles(smile)
        if mol:
            atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            num_carbons = atoms.count(6)
            num_nitrogens = atoms.count(7)
            num_oxygens = atoms.count(8)
            num_sulpurs = atoms.count(16)
            num_chlorine = atoms.count(17)
            return [num_carbons, num_nitrogens, num_oxygens, num_sulpurs, num_chlorine]

#need to take a file list of smiles and retuen list of encoded vectors
#could extra function for file reading

#file processing can be used for building training set
#list evaluation can be used for objective evalauation