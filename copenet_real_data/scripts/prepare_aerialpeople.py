
import os.path as osp
import subprocess
from tqdm import tqdm
import os
import glob
import pickle as pkl
import shutil
import cv2

#%% create cropped dataset
num_cams = 4
dataset_root = "/ps/project/datasets/AirCap_ICCV19/agora_unreal_4views"

final_dataset_root = "/ps/project/datasets/AirCap_ICCV19/agora_unreal_4views_cropped"

os.mkdir(final_dataset_root)

# shutil.copytree(osp.join(dataset_root,"dataset"),osp.join(final_dataset_root,"dataset"))
subprocess.call(["rsync","-av",osp.join(dataset_root,"dataset"),osp.join(final_dataset_root)])

os.mkdir(osp.join(final_dataset_root,"data"))
for sample_name in tqdm(glob.glob(osp.join(dataset_root,"dataset","pkls","*"))):

    sub = sample_name.split("/")[-1]

    os.mkdir(osp.join(final_dataset_root,"data",sub))


for sample_name in tqdm(glob.glob(osp.join(dataset_root,"dataset","pkls","*","*.pkl"))):
    sample = pkl.load(open(sample_name,"rb"))

    sub = sample_name.split("/")[-2]
    pose_suffix = sample_name.split("/")[-1].split(".")[0].split("_")[-1]


    # try to randomly add to BB in all the direction; use BB if goes out of image
    r = 200
    for i in range(num_cams):
        ymin = (sample['bb'+str(i)][0][1] - r) if (sample['bb'+str(i)][0][1] - r) > 0 else 0
        ymax = (sample['bb'+str(i)][1][1] + r) if (sample['bb'+str(i)][1][1] + r) < 1080 else 1080
        xmin = (sample['bb'+str(i)][0][0] - r) if (sample['bb'+str(i)][0][0] - r) > 0 else 0
        xmax = (sample['bb'+str(i)][1][0] + r) if (sample['bb'+str(i)][1][0] + r) < 1920 else 1920 

        im = cv2.imread(osp.join(dataset_root,sample["im"+str(i)]))
        cv2.imwrite(osp.join(final_dataset_root,"data",sub,"MyCamera"+str(i)+"_"+pose_suffix+".png"),im[ymin:ymax,xmin:xmax])
        
    # import ipdb; ipdb.set_trace()



#%% Train and test data split
import numpy as np
import os.path as osp
from tqdm import tqdm
import os
import glob
import pickle as pkl
import shutil
import cv2


samples = sorted(os.listdir(osp.join(final_dataset_root,"dataset","pkls")))
all_pkls = sorted(glob.glob(osp.join(final_dataset_root,"dataset","pkls","*","*.pkl")))
subs = np.unique([x.split("_")[1] for x in samples])

test_subs = subs[-10:]
train_subs = subs[:-10]

test_pkls = []
for sub in test_subs:
    test_pkls += [x for x in all_pkls if (sub == x.split("/")[-1].split("_")[1])]

train_pkls = []
for sub in train_subs:
    train_pkls += [x for x in all_pkls if (sub == x.split("/")[-1].split("_")[1])]

pkl.dump(train_pkls,open(osp.join(final_dataset_root,"dataset","train_pkls.pkl"),"wb"))
pkl.dump(test_pkls,open(osp.join(final_dataset_root,"dataset","test_pkls.pkl"),"wb"))
