#!/usr/bin/env python
# coding: utf-8

# ## Spacenet
# 
# Prediction using saved model
# 




# In[2]:


import datetime
import uuid


# In[3]:


import PIL.Image as pil_image


# In[4]:


from fastai import *
from fastai.vision import *


# In[5]:


from fastai.utils import *
import fastai
print(fastai.__version__)


# In[6]:


torch.cuda.set_device(1)


# #### saved model ids

# In[7]:


DATE = '20181209'
print(f'DATE: {DATE}') 


# In[8]:


UID='af27e193'
print(f'UID: {UID}') 


# In[9]:


ARCH = models.resnet34
ARCH_NAME = 'rn34'
MODEL_NAME = 'unet'


# In[10]:


src_size=(450,450)


# ## Load Data

# In[11]:


path = Path('../../ssd_data/SpaceNet_Off-Nadir_Dataset')
path.ls()


# In[12]:


path_img = path/'resized_train'
path_lbl = path/'resized_masks'
path_test = path/'resized_test'


# In[40]:


fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)
test_fnames = get_image_files(path_test)
fnames[:3], lbl_names[:3] 


# In[14]:


len(fnames), len(lbl_names)


# In[15]:


def get_y_fn(full_name):
    parts = str(full_name).split('_')
    im_id = '_'.join(parts[-2:])
    mask_name = 'mask_'+im_id
    return path_lbl/f'{mask_name}'


# # Make DataBunch

# In[16]:


size = src_size
bs=4


# In[17]:


holdout_grids = ['735851','747551','741251','746201']
valid_idx = [i for i,o in enumerate(fnames) if any(c in str(o) for c in holdout_grids)]


# In[18]:


codes = np.array(['nadir','offnadir','faroffnadir'])


# In[19]:


# subclassing SegmentationLabelList to set open_mask(fn, div=True), probably a better way to do this?
# idea from https://forums.fast.ai/t/unet-binary-segmentation/29833/40

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)
    
class SegItemListCustom(ImageItemList):
    _label_cls = SegLabelListCustom


# In[20]:


src = (SegItemListCustom.from_folder(path_img)
        #.split_by_idx(valid_idx)
        .random_split_by_pct(0.2)
        .label_from_func(get_y_fn, classes=codes))


# In[21]:


tfms = get_transforms(flip_vert=True, max_warp=0, max_zoom=1.2, max_lighting=0.3)
data = (src.transform(tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[22]:


data.export


# In[23]:


data.valid_ds.items


# In[24]:


data.train_ds.y[1].data


# In[25]:


data.classes


# ### data load

# In[26]:


# DataBunch.load_empty(path) (where path points to where export.pkl file is)
export_path = path_img


# In[27]:


empty_data = data.load_empty(export_path, tfms=tfms, tfm_y=True, size=src_size).normalize(imagenet_stats)


# # Custom Loss

# In[28]:


import pdb


# In[29]:


def dice_loss(input, target):
#     pdb.set_trace()
    smooth = 1.
    input = input[:,1,None].sigmoid()
    iflat = input.contiguous().view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()
    return (1 - ((2. * intersection + smooth) / ((iflat + tflat).sum() +smooth)))

def combo_loss(pred, targ):
    bce_loss = CrossEntropyFlat(axis=1)
    return bce_loss(pred,targ) + dice_loss(pred,targ)


# ## Define Model

# In[30]:


def acc_fixed(input, targs):
    n = targs.shape[0]
    targs = targs.squeeze(1)
    targs = targs.view(n,-1)
    input = input.argmax(dim=1).view(n,-1)
    return (input==targs).float().mean()

def acc_thresh(input:Tensor, target:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    
#     pdb.set_trace()
    if sigmoid: input = input.sigmoid()
    n = input.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    target = target.view(n,-1)
    return ((input>thresh)==target.byte()).float().mean()


# In[31]:


# iou = partial(dice, iou=True)
metrics = [dice_loss, acc_thresh, dice]


# In[32]:


learn = unet_learner(empty_data, ARCH, metrics=metrics)


# In[33]:


#best at this point in time
learn.load(f'{DATE}-{ARCH_NAME}-{MODEL_NAME}-comboloss-best-{UID}')
learn.model.eval()
learn.


# In[59]:


img = open_image(test_fnames[600])
img.show()
plt.show()


# In[60]:


pred_c,pred_idx,outputs=learn.predict(img)


# In[61]:


pred_c.show()


# In[ ]:


pred_c.save()

