import torch
import numpy as np
from rdkit import Chem
import subprocess
import pandas

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

def get_headings():
    """
    This returns a fixed order for descriptor headings, as DescriptorCalculator may give back vectors in any order.
    :return:
    """
    headings = 'logP,Vx,MW,NegativeCharge,PositiveCharge,Flex,AromaticRings,OverallCharge,ERTLNotPSA,ERTLNoSPtPSA,HBA-lip,HBA-prof,HBD-lip,HBD-prof,HBD-cam,quatN,macrocyclic,ACamideO-nh-nh2,ACamideO-nh0,ASamideO-nh-nh2,ASamideO-nh0,Aamidine,AbasicNH0,AbasicNH1,CBr,CF3,CH0Aa,CH1Aa,CH2Aa,CH2hetero,CH2link,CH2long,CH3Aa,CH3hetero,CSSC,CamideNH0,Ester,HaloC,Michael-accept,NBA,NH1and2CdO,NO,NRB,OHCHCdO,Ocarbamate,Pamide,Pester,RCamideO-nh-nh2,RCamideO-nh0,RSR,RSamideO-nh-nh2,RSamideO-nh0,Ramidine,RbasicNH0,RbasicNH1,Samide-NH,SamideNH0,activatedCH,aldehydes,aliphOH-t6,allylic-oxyd-t10,amide-dicarbonyl,aminoethanol0,aminoethanol1,anycarbonyl,aromBr,aromCl,aromF,aromI,aromO,arylNHCO,basic-NH2,benzdiaz-t18,benzdiazepine-ring,benzylicOH,branchedCnotRing,carbamate-and-thio,carbonate-carbamate,ch2-lipo-t9,dNO,di-widhraw-cx4,diazo-aryl,diazo,dione-1-4,easy-oxy-t13,ertl-33,ertl-35,ertl-37,ertl-39,ertl-41,ertl-43,est-lact-latm-carbm-t7,ether,halosp3sp3halo,hetero-halo-di-n-arom,hindred-phenol,hydroxyA,hydroxylation-t8,intraHbond5,intraHbond6,ketal,ketone-t14,ketones,lipovolume,nH0indole-like,nHindole-like,nc(do)n,nitro-O,nitro-no-ortho-t15,nitro,nonring-at,not-ring-diol,ohccn-t17,p-hetero-or-halo,p-withdraw-phenol,perfluoro,phenol-pyr2r,phenol,phenolic-tautomer,poly-sugars,polyOH,pyridine,pyridones,quinone-type,ring-join,ring5-nH0,ring5nH,ringOdouble,ringat,ringdiol,sp-carbons,sp2-carbons,spiroC,sulfonicacid,sulphates,sulphonamide-t5,t-16-1,t-16-2,t-16-3,tert-amine-t11,thio-acid,thio-keto,urea-thio,urea,xccn-t12,zw1,zw2,zw3,nC(sp2),nC(sp3),nCOOH,nOH,nCO,nOS,nX,nNprot,dCH2,ssCH2,tCH,dsCH,aaCH,sssCH,ddC,tsC,dssC,aasC,aaaC,ssssC,sNH3+,sNH2,ssNH2+,dNH,ssNH,aaNH,tN,sssNH+,dsN,aaN,sssN,ddsN,aasN,ssssN+,sOH,ssO,sF,sSiH3,ssSiH2,sssSiH,ssssSi,sPH2,ssPH,sssP,dsssP,sssssP,sSH,dS,ssS,aaS,dssS,ddssS,sCl,sBr,sI,nNneutral,NnH,N4,NbN,fg5,CamideNH,BasicNH0R2AroRings,BasicNH02AroRings,BasicNH1R2AroRings,BasicNH12AroRings,NonOrganicAtom,PRX-time1,PRX-time-1,UB,HDN,HAN,PRX-time2,HAS,HAT,HAO,AliRingAttachment,C12,C4,C10,C6,C3,C9,C8,C1,C11,C2,C27,C26,N6,N7,N8,N14,N2,N13,H3,N1,BasicGroup,N10,HDT,HDO,AcidGroup,H4,H2,O7,O6,O3,O11,O5,O9,O10,S2,AroRingAttachment,C25,C13,N11,N12,HydrophobicGroup,H1a,C5,C21,C22,C23,C24,C20,S3,ed70,ed20,ed50,ed60,ed40,ed80,ew70,ew60,ew90,ew80,ew75,ew30,ew50,ew40,ew20,ew10,ew100,f004,f005,f007,f015,f147,f244,f245,f301,f390,f392,f393,f407,f413,f440,f441,f443,f444,f456,q017,q039,q040,q041,q137,q139,q155,q192,q257,q277,q300,q358,q453,q457,q458,q481,q483,q485,frg-8,frg-26,frg-54,Nn'
    headings = headings.split(',')
    return headings

def get_moments(vectors):
    vectors = np.array(vectors).transpose()
    mew = [np.average(x) for x in vectors]
    std = [np.std(x) for x in vectors]
    return mew, std

def get_latent_vector(smiles, vec_file=None, moments=False):
    headings = get_headings()

    if vec_file is None:
        generated_vectors = True
        #check of smiles are valid
        # write smiles to file
        invalid = []
        with open('./current_smi.dat', 'w') as f:
            f.write('INDEX, SMILES\n')
            for i, smi in enumerate(smiles):
                f.write('{0}, {1}\n'.format(smi, i))
                if not Chem.MolFromSmiles(smi):
                    # record indexes of bad smiles
                    invalid.append(i)
        print('{} invalid smiles caught'.format(len(invalid)))

        #run descriptor generation, last 2 arguments are silencing this subprocess
        subprocess.call(["/usr/bin/python2.7", "./DescriptorCalculation.py"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        vec_file = './results/output'
    else:
        generated_vectors = False

    #reading descriptors
    data = pandas.read_csv(vec_file)
    # correct heading order
    data = data.reindex(columns=headings)
    data = data.values

    #Address bad vectors in generated data, if we have generated them from smiles internally
    if generated_vectors:
        data = [x if i not in invalid else False for i, x in enumerate(data)]

    #Calculate mew and std for each colum, used in scoring function to normalize data
    if moments:
        vectors = data.transpose()
        mew = [np.average(x) for x in vectors]
        std = [np.std(x) for x in vectors]
        #catch any zeros which will give nan when normalizing
        std = [x if x != 0 else 1.0 for x in std]
        return data, mew, std
    else:
        return data


