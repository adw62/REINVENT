#!/usr/local/bin/python
import sys
import csv

#sys.path.append("/usr/local/lib/python2.6/dist-packages/")

import AmgCalc
import os
import struct

smiles = []
smarts = []
compids = []
logfile = ''

"""
class DescpCalc(object):
    def __init__(self, filename):
        yvalues = extractData(filename)
"""





#the AmgCalc type codes - make changes here if AmgCalc changes
SPLIT = 1

LAST_WORK_TYPE = 10


def extractData(filename):
    global yvalues
    try:
        with open(filename) as input_file:
            reader = csv.DictReader(input_file)
            cols = reader.fieldnames
            for entry in reader:
                #now iterate over the rows of data to build the lists
                smiles.append(entry[cols[0]])
                compids.append(entry[cols[1]])
        #check for duplicates
        if len(compids) != len(set(compids)):
            print(compids)
            print('Non unique ids used')
            return False
        yvalues = [0.0 for s in smiles]
    except Exception as e:
        print(e)
        return False
    return True

def loadSmarts():
    global smarts
    try:
        f = open('AMGSmarts.txt', 'r')
    except(IOError, ioex):
        print('Error opening smarts file: ',ioex)
        return False
    #check the number of columns
    rows = f.readlines()

    for r in rows:
        r = r.rstrip()
        entry = r.split(' ')
        smarts.append(entry)


def getSelectionParams():
    trn = 0.70
    val = 0.15
    minsd = 0.000
    minocc = 0.0
    maxcorr = 1.1#Set to > 1 to ensure no descriptors get omitted
    tani = 0.7   
    splitTech = 2#Random?
    index1=0
    index2=1

    return (trn,val,minsd,minocc,maxcorr,tani,splitTech,index1,index2)

def splitSets():
        selparams = getSelectionParams()
        print('creating object')
        calc = AmgCalc.WorkObject(SPLIT)
        calc.setPath("./")
        print('created object')
        calc.loadData(yvalues,smiles, compids, smarts, [], [])
        print('loaded data')
        calc.setParams(selparams)
        print('set params')
        return calc.doCalc()

def getSets(splitres):
    trnsmiles = []
    valsmiles = []
    testsmiles = []
    trnvals = []
    valvals = []
    testvals = []
    trnids = []
    valids = []
    testids = []
    trndescvals = []
    valdescvals = []
    testdescvals = []
    for i in splitres[0]:
        trnsmiles.append(smiles[i])
        trnids.append(compids[i])
        trnvals.append(yvalues[i])
        trndescvals.append(splitres[5][i])
    for i in splitres[1]:
        valsmiles.append(smiles[i])
        valids.append(compids[i])
        valvals.append(yvalues[i])
        valdescvals.append(splitres[5][i])
    for i in splitres[2]:
        testsmiles.append(smiles[i])
        testids.append(compids[i])
        testvals.append(yvalues[i])
        testdescvals.append(splitres[5][i])
    return (trnsmiles,valsmiles,testsmiles,trnids,valids,testids,trnvals,valvals,testvals,trndescvals,valdescvals,testdescvals)

if __name__ == "__main__":

        #if len(sys.argv) != 2:
        #    print('Usage: DescriptorCalculation.py <filename> ')
        #    os._exit(-1)


        #extract the required data from the file
        ok = extractData('./current_smi.dat')
        if ok == False:
            os._exit(-1)

        #load the smarts
        loadSmarts()

        #create a folder for results if not already present
        try:
            os.mkdir('results')
        except:
            pass

        
        splitres = splitSets()
        trnsmiles,valsmiles,testsmiles,trnids,valids,testids,trnvals,valvals,testvals,trndescvals,valdescvals,testdescvals = getSets(splitres)
        results = {}
        with open('results/output', 'w') as output_file:
           descs = [desc[1] for desc in smarts]
           writer = csv.writer(output_file)
           writer.writerow(["CompoundID"] + descs)
           for i in range(len(trnsmiles)):
                results[trnids[i]] =  trndescvals[i]
           for i in range(len(valsmiles)):
                results[valids[i]] =  valdescvals[i]
           for i in range(len(testsmiles)):
                results[testids[i]] =  testdescvals[i]
           for compound in compids:
                writer.writerow([compound] +['%f' % v for v in results[compound]])


 
 
