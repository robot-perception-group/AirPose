import pickle as pkl
import sys
import os

data_root = sys.argv[1]

train_ds = pkl.load(open(os.path.join(data_root,"dataset","train_pkls.pkl"),"rb"))
test_ds = pkl.load(open(os.path.join(data_root,"dataset","test_pkls.pkl"),"rb"))

train_ds = [os.path.join(data_root,*x.split("/")[-4:]) for x in train_ds]
test_ds = [os.path.join(data_root,*x.split("/")[-4:]) for x in test_ds]

pkl.dump(train_ds,open(os.path.join(data_root,"dataset","train_pkls.pkl"),"wb"))
pkl.dump(train_ds,open(os.path.join(data_root,"dataset","test_pkls.pkl"),"wb"))

print("done!!!")