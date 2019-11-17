import sys

print("SMILES,INDEX")
idx = 0 
with open(sys.argv[1]) as f:
  for line in f:
    print(line.strip() + "," + str(idx))
    idx+=1
