import torch
import numpy as np

import rdkit
from rdkit import six
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Fragments

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def get_latent_vector(smiles_string, pad = 300):
    """
    :param smiles_string: SMILES string of the compound under study - string.
    :return: Returns a numpy array of descriptors.
    """

    if Chem.MolFromSmiles(smiles_string):
        vector = np.array([Descriptors.MolWt(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcLabuteASA(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumAliphaticCarbocycles(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumAliphaticHeterocycles(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumAliphaticRings(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumAmideBonds(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumAromaticCarbocycles(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumAromaticHeterocycles(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumHBA(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumHBD(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumHeteroatoms(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumHeterocycles(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcTPSA(Chem.MolFromSmiles(smiles_string)),
        rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(smiles_string))[0],
        rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(smiles_string))[1],
        Fragments.fr_Al_COO(Chem.MolFromSmiles(smiles_string)),
        Fragments.fr_Al_OH(Chem.MolFromSmiles(smiles_string)),
        Fragments.fr_Ar_COO(Chem.MolFromSmiles(smiles_string)),
        Fragments.fr_Ar_OH(Chem.MolFromSmiles(smiles_string)),
        Fragments.fr_COO(Chem.MolFromSmiles(smiles_string)),
        Fragments.fr_C_O(Chem.MolFromSmiles(smiles_string)),
        Fragments.fr_benzene(Chem.MolFromSmiles(smiles_string)),
        Fragments.fr_amide(Chem.MolFromSmiles(smiles_string))])
    else:
        vector = np.ones(26)*1000 #dummy vector with correct lenght will score badly
    return np.append(vector, np.zeros(pad))


