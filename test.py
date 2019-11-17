
#!/usr/bin/env python

import torch
import pickle
import numpy as np
import time
import os
from shutil import copyfile
import math
from rdkit import Chem
from rdkit import rdBase

from model import RNN
from data_structs import Vocabulary
from torch.utils.data import DataLoader
from data_structs import Dataset
from utils import seq_to_smiles, get_latent_vector

testing_vector = False

#order of headings used in training
headings = 'logP,Vx,MW,NegativeCharge,PositiveCharge,Flex,AromaticRings,OverallCharge,ERTLNotPSA,ERTLNoSPtPSA,HBA-lip,HBA-prof,HBD-lip,HBD-prof,HBD-cam,quatN,macrocyclic,ACamideO-nh-nh2,ACamideO-nh0,ASamideO-nh-nh2,ASamideO-nh0,Aamidine,AbasicNH0,AbasicNH1,CBr,CF3,CH0Aa,CH1Aa,CH2Aa,CH2hetero,CH2link,CH2long,CH3Aa,CH3hetero,CSSC,CamideNH0,Ester,HaloC,Michael-accept,NBA,NH1and2CdO,NO,NRB,OHCHCdO,Ocarbamate,Pamide,Pester,RCamideO-nh-nh2,RCamideO-nh0,RSR,RSamideO-nh-nh2,RSamideO-nh0,Ramidine,RbasicNH0,RbasicNH1,Samide-NH,SamideNH0,activatedCH,aldehydes,aliphOH-t6,allylic-oxyd-t10,amide-dicarbonyl,aminoethanol0,aminoethanol1,anycarbonyl,aromBr,aromCl,aromF,aromI,aromO,arylNHCO,basic-NH2,benzdiaz-t18,benzdiazepine-ring,benzylicOH,branchedCnotRing,carbamate-and-thio,carbonate-carbamate,ch2-lipo-t9,dNO,di-widhraw-cx4,diazo-aryl,diazo,dione-1-4,easy-oxy-t13,ertl-33,ertl-35,ertl-37,ertl-39,ertl-41,ertl-43,est-lact-latm-carbm-t7,ether,halosp3sp3halo,hetero-halo-di-n-arom,hindred-phenol,hydroxyA,hydroxylation-t8,intraHbond5,intraHbond6,ketal,ketone-t14,ketones,lipovolume,nH0indole-like,nHindole-like,nc(do)n,nitro-O,nitro-no-ortho-t15,nitro,nonring-at,not-ring-diol,ohccn-t17,p-hetero-or-halo,p-withdraw-phenol,perfluoro,phenol-pyr2r,phenol,phenolic-tautomer,poly-sugars,polyOH,pyridine,pyridones,quinone-type,ring-join,ring5-nH0,ring5nH,ringOdouble,ringat,ringdiol,sp-carbons,sp2-carbons,spiroC,sulfonicacid,sulphates,sulphonamide-t5,t-16-1,t-16-2,t-16-3,tert-amine-t11,thio-acid,thio-keto,urea-thio,urea,xccn-t12,zw1,zw2,zw3,nC(sp2),nC(sp3),nCOOH,nOH,nCO,nOS,nX,nNprot,dCH2,ssCH2,tCH,dsCH,aaCH,sssCH,ddC,tsC,dssC,aasC,aaaC,ssssC,sNH3+,sNH2,ssNH2+,dNH,ssNH,aaNH,tN,sssNH+,dsN,aaN,sssN,ddsN,aasN,ssssN+,sOH,ssO,sF,sSiH3,ssSiH2,sssSiH,ssssSi,sPH2,ssPH,sssP,dsssP,sssssP,sSH,dS,ssS,aaS,dssS,ddssS,sCl,sBr,sI,nNneutral,NnH,N4,NbN,fg5,CamideNH,BasicNH0R2AroRings,BasicNH02AroRings,BasicNH1R2AroRings,BasicNH12AroRings,NonOrganicAtom,PRX-time1,PRX-time-1,UB,HDN,HAN,PRX-time2,HAS,HAT,HAO,AliRingAttachment,C12,C4,C10,C6,C3,C9,C8,C1,C11,C2,C27,C26,N6,N7,N8,N14,N2,N13,H3,N1,BasicGroup,N10,HDT,HDO,AcidGroup,H4,H2,O7,O6,O3,O11,O5,O9,O10,S2,AroRingAttachment,C25,C13,N11,N12,HydrophobicGroup,H1a,C5,C21,C22,C23,C24,C20,S3,ed70,ed20,ed50,ed60,ed40,ed80,ew70,ew60,ew90,ew80,ew75,ew30,ew50,ew40,ew20,ew10,ew100,f004,f005,f007,f015,f147,f244,f245,f301,f390,f392,f393,f407,f413,f440,f441,f443,f444,f456,q017,q039,q040,q041,q137,q139,q155,q192,q257,q277,q300,q358,q453,q457,q458,q481,q483,q485,frg-8,frg-26,frg-54,Nn'
headings = headings.split(',')

def black_box(load_weights='data/Prior.ckpt', batch_size=1):

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc")

    if testing_vector:
        vec_file = None
    else:
        vec_file = "data/test_dat/vecs.dat"

        data = {}
        with open(vec_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    keys = (line.strip('\n')).split(',')
                else:
                    vector = (line.strip('\n')).split(',')
                    for k, v in zip(keys, vector):
                        if k in data:
                            data[k].append(float(v))
                        else:
                            data[k] = [float(v)]

        data = [data[k] for k in headings]
        data = [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]
        data = torch.FloatTensor(data)

    voc = Vocabulary(init_from_file="data/Voc")
    Prior = RNN(voc, len(data[0]))

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(load_weights))
    else:
        Prior.rnn.load_state_dict(torch.load(load_weights, map_location=lambda storage, loc: storage))


    for test_vec in data:
        
        print('Test vector {}'.format(test_vec))
        test_vec = test_vec.float()
        valid = 0
        num_smi = 100
        all_smi = []
        for i in range(num_smi):
            seqs, prior_likelihood, entropy = Prior.sample(batch_size, test_vec)
            smiles = seq_to_smiles(seqs, voc)[0]
            if Chem.MolFromSmiles(smiles):
                        valid += 1
                        all_smi.append(smiles)
        if testing_vector:
            print('Output smiles and vector {0}, {1}'.format(smiles,  get_latent_vector(smiles)))
        else:
            for smi in all_smi:
                print(smi)
        print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(range(num_smi))))

black_box()
