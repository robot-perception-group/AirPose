# %%
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from tqdm import tqdm
import pickle as pkl
import os
import matplotlib.pyplot as plt
import cv2
os.environ["PYOPENGL_PLATFORM"] = 'egl'


from copenet_real.copenet_twoview import copenet_twoview
from copenet_real.dsets import aircapData

ckpt_path = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch=761.ckpt"
# ckpt_path = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/rebuttal_aircapdata/rebuttal3/checkpoints/epoch=210.ckpt"
# check model type
model_type = ckpt_path.split("/")[-4]

# create trainer
trainer = Trainer(gpus=1)
# create Network


net = copenet_twoview.load_from_checkpoint(checkpoint_path=ckpt_path).to("cuda")

# create dataset and dataloader
train_ds, test_ds = aircapData.get_copenet_real_traintest()

tst_dl = DataLoader(test_ds, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)
trn_dl = DataLoader(train_ds, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)

batch = iter(trn_dl).next()
for k in batch:
    try:
        batch[k] = batch[k].to("cuda")
    except:
        pass
output,losses,_ = net.fwd_pass_and_loss(batch)

pred_im,in_im = net.summaries(batch,output,losses,is_test=True)
plt.imshow(pred_im.permute(1,2,0).detach().cpu().numpy())
cv2.imwrite("airpose_on_aircap_test.png",pred_im.permute(1,2,0).detach().cpu().numpy()*255)



# %%
