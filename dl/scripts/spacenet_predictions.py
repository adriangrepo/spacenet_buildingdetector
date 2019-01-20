#!/usr/bin/env python
# coding: utf-8

# ## Spacenet
# 
# https://medium.com/the-downlinq/establishing-a-machine-learning-workflow-530628cfe67
# 
# https://medium.com/the-downlinq/object-detection-on-spacenet-5e691961d257
# 
# https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb
# 
# https://medium.com/the-downlinq/a-baseline-model-for-the-spacenet-4-off-nadir-building-detection-challenge-6b7983312b4b
# 
# https://medium.com/the-downlinq
# 

# In[35]:


# In[36]:


import datetime
import uuid
import glob


# In[37]:


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


# In[38]:


import PIL.Image as pil_image


# In[39]:


import fastai
print(fastai.__version__)


# In[40]:


torch.cuda.set_device(0)


# In[41]:


#DATE = datetime.datetime.today().strftime('%Y%m%d')
DATE='20181204'
print(f'DATE: {DATE}') 


# In[42]:


#UID=str(uuid.uuid4())[:8]
UID='3d4a81ea'
print(f'UID: {UID}') 


# In[43]:


ARCH = models.resnet34
ARCH_NAME = 'rn34'
MODEL_NAME = 'unet'


# In[44]:


SUB_NUM='1'


# In[45]:


path = Path('../../../data/SpaceNet_Off-Nadir_Dataset')
path.ls()


# In[46]:


path_img = path/'resized_train'
path_lbl = path/'resized_masks'
path_test = path/'resized_test'


# In[47]:


fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)
test_fnames = get_image_files(path_test)
fnames[:3], lbl_names[:3] 


# In[48]:


len(fnames), len(lbl_names), len(test_fnames)


# In[49]:


#only 1064 masks and images - but multiple nadirs


# In[50]:


image_ids =[]
image_names=[]
channel_types=[]
nadir_angles=[]
mask_names=[]
nadir_types=[]
for n in fnames:
    parts = str(n).split('_')
    im_id = '_'.join(parts[-2:])
    image_ids.append(im_id)
    indici = [i for i, s in enumerate(parts) if 'nadir' in s]
    indici=indici[0]
    
    nadir_angle = parts[indici].split('nadir')[1]
    nadir_angles.append(nadir_angle)
    if int(nadir_angle) < 26:
        angle_set = 'nadir'
    elif int(nadir_angle) > 25 and int(nadir_angle) < 40:
        angle_set = 'offnadir'
    elif int(nadir_angle) > 40:
        angle_set = 'faroffnadir'
    nadir_types.append(angle_set)
    
    fname_part = str(n).split('/')[-1]
    image_names.append(fname_part)
    
    channel_type = str(fname_part).split('Atlanta')[0][:-1]
    channel_types.append(channel_type)
    
    mask_name = 'mask_'+im_id
    mask_names.append(mask_name)


# In[51]:


len(list(set(image_ids)))


# In[52]:


train_df = pd.DataFrame(
    {'image_name': image_names,
     'channel_type': channel_types,
     'nadir_angle': nadir_angles,
     'nadir_type': nadir_types,
     'mask_name': mask_names
    })


# In[53]:


train_df.head()


# In[54]:


train_df['channel_type'].unique()


# In[55]:


train_df['nadir_angle'].unique()


# In[56]:


#### resample example


# In[57]:


def show_resized():
    img_f = train_df['image_name'][0]
    mask_n = train_df['mask_name'][0]
    img = open_image(RESIZED_TRAIN/f'{img_f}')
    mask = open_mask(RESIZED_MASKS/f'{mask_n}', div=True)

    fig,ax = plt.subplots(1,1, figsize=(10,10))
    img.show(ax=ax)
    mask.show(ax=ax, alpha=0.5)


# In[58]:


#### original


# In[59]:


def show_original():
    img_f = train_df['image_name'][0]
    mask_n = train_df['mask_name'][0]
    img_f=img_f.split('.png')[0]
    img_f=img_f+'.tif'
    mask_n=mask_n.split('.png')[0]
    mask_n=mask_n+'.tif'
    img = open_image(path_img/f'{img_f}')
    mask = open_mask(path_lbl/f'{mask_n}', div=True)
    
    src_size = np.array(mask.shape[1:])
    print(src_size)
    print(mask.data)

    fig,ax = plt.subplots(1,1, figsize=(10,10))
    img.show(ax=ax)
    mask.show(ax=ax, alpha=0.5)


# In[60]:


mask_n = train_df['mask_name'][0]
mask = open_mask(path_lbl/f'{mask_n}', div=True)
src_size = np.array(mask.shape[1:])


# ## Preds

# https://spacenetchallenge.github.io/datasets/spacenet-OffNadir-summary.html
# 
# In the SpaceNet Off-Nadir Building Extraction Challenge, the metric for ranking entries is the SpaceNet Metric.
# This metric is an F1-Score based on the intersection over union of two building footprints with a threshold of 0.5
# 
# F1-Score is calculated by taking the total True Positives, False Positives, and False Negatives for each nadir segement and then averaging the F1-Score for each segement.
# 
# F1-Score Total = mean(F1-Score-Nadir, F1-Score-Off-Nadir, F1-Score-Very-Off-Nadir)

# Your output must be a CSV file with almost identical format to the building footprint definition files.
# 
# ImageId,BuildingId,PolygonWKT_Pix,Confidence
# 
# Your output file may or may not include the above header line. The rest of the lines should specify the buildings your algorithm extracted, one per line.
# 
# The required fields are:
# 
# ImageId is a string that uniquely identifies the image.
# BuildingId is an integer that identifies a building in the image, it should be unique within an image and must be positive unless the special id of -1 is used. -1 must be used to signal that there are no buildings in the image.
# PolygonWKT_Pix specifies the points of the shape that represents the building you found. The format is exactly the same as given above in the Input files section. Important to know that the coordinates must be given in the scale of the 3-band images. So if you find a building that has a corner at (40, 20) on the 3-band image and (10, 5) on the corresponding 8-band image then your output file should have a (40 20 0) coordinate triplet listed in the shape definition.
# Confidence is a positive real number, higher numbers mean you are more confident that this building is indeed present. See the details of scoring for how this value is used.
# Your output must be a single file with .csv extension. Optionally the file may be zipped, in which case it must have .zip extension. The file must not be larger than 150MB and must not contain more than 2 million lines.
# 
# Your algorithm must process the image tiles of the test set one by one, that is when you are predicting building footprints you must not use information from other tiles of the test set.
# 

# In[ ]:





# In[73]:


bs=4


# In[66]:


def get_y_fn(full_name):
    parts = str(full_name).split('_')
    im_id = '_'.join(parts[-2:])
    mask_name = 'mask_'+im_id
    return path_lbl/f'{mask_name}'


# In[67]:


codes = np.array(['nadir','offnadir','faroffnadir'])


# In[68]:


holdout_grids = ['735851','747551','741251','746201']
valid_idx = [i for i,o in enumerate(fnames) if any(c in str(o) for c in holdout_grids)]


# In[88]:


#here we default tpo using codes instad of None for classes - otherwise too much work overriding methods
#when do load_enpty()
class SegmentationLabelList(ImageItemList):
    def __init__(self, items:Iterator, classes:Collection=codes, **kwargs):
        super().__init__(items, **kwargs)
        self.classes,self.loss_func = classes,CrossEntropyFlat()
        self.c = len(self.classes)

    def new(self, items, classes=None, **kwargs):
        return self.__class__(items, ifnone(classes, self.classes), **kwargs)

    def open(self, fn): return open_mask(fn, div=True)
    
class SegmentationItemList(ImageItemList): _label_cls = SegmentationLabelList


# In[89]:


#recreate data via method when training
src = (SegmentationItemList.from_folder(path_img)
        .split_by_idx(valid_idx)
        .label_from_func(get_y_fn, classes=codes))


# In[90]:


print(type(src))


# In[91]:


print(src.classes)


# In[92]:


tfms = get_transforms(flip_vert=True, max_warp=0, max_zoom=1.2, max_lighting=0.3)
data = (src.transform(tfms, size=src_size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[93]:


print(data.classes)


# In[94]:


tfms = get_transforms(flip_vert=True, max_warp=0, max_zoom=1.2, max_lighting=0.3)


# In[95]:


# DataBunch.load_empty(path) (where path points to where export.pkl file is)
export_path = path_img


# In[96]:


print(export_path)


# In[97]:


#DataBunch.load_empty = _databunch_load_empty


# In[98]:


print(type(data))
#<class 'fastai.vision.data.ImageDataBunch'>


# In[99]:


#data at this point:
#ImageDataBunch(DataBunch)
#    train_dl
#        dl
#            dataset
#                c
#    valid_dl
#        dl
#            dataset
#                c


# In[100]:


#ImageDataBunch(DataBunch)
#                  Databunch.load_empty()
#                                 @classmethod
#                                 def _databunch_load_empty(cls, path, fname:str='export.pkl', tfms:TfmList=None, tfm_y:bool=False, **kwargs):
#                                      "Load an empty `DataBunch` from the exported file in `path/fname` with optional `tfms`."
#                                     ds = LabelList(Dataset).load_empty(path/fname, tfms=(None if tfms is None else tfms[1]), tfm_y=tfm_y, **kwargs)
#                                     return cls.create(ds,ds,path=path)


# In[101]:


#here we now reset to empty

#LabelList(Dataset).load_empty


# In[102]:


empty_data = data.load_empty(export_path, tfms=tfms, tfm_y=True, size=src_size).normalize(imagenet_stats)


# In[103]:


iou = partial(dice, iou=True)
metrics = [iou, dice]


# In[104]:


#LabelList(Dataset)
#    @property
#    c()
#
#    @classmethod
#    def load_empty(cls, fn:PathOrStr, tfms:TfmList=None, tfm_y:bool=False, **kwargs):


# In[105]:


#train data

#here data==ImageDataBunch->DataLoader
#SegmentationItemList(ImageItemList)
#    _label_cls = SegmentationLabelList
#                       SegmentationLabelList(ImageItemList)
#                                                  ImageItemList(ItemList)
#                                                      _bunch=ImageDataBunch
#                                                                ImageDataBunch(DataBunch)
#                                                                                 train_dl {DeviceDataLoader} 
#                                                                                               dl {DataLoader}
#                                                                                                      dataset {LabelList(Dataset)}
#                                                                                                                           y {SegmentationLabelList}
#                                                                                                                           @property
#                                                                                                                           c


# In[106]:


#empty_data


#here data==ImageDataBunch->DataLoader

#ImageDataBunch(DataBunch)
#                 train_dl {DeviceDataLoader} 
#                               dl {DataLoader}
#                                       dataset {LabelList(Dataset)}
#                                                             y {SegmentationLabelList(ImageItemList)}
#
#                                                             @property


# In[107]:


learn = unet_learner(empty_data, ARCH, metrics=metrics)


# In[108]:


learn.load(f'{DATE}-{ARCH_NAME}-{MODEL_NAME}-stage2_2')
learn.model.eval()


# In[115]:


def pred_images(test_file_names):
    #im_files = glob.glob(f"{test_path}/*.png")
    i=0
    for fname in test_file_names:
        # load image and predict
        img = open_image(fname)
        pred_class, pred_idx, outputs = learn.predict(img)
        if i<10:
            display(img); pred_class; outputs
        i+=1


# In[116]:


pred_images(test_fnames)


# In[117]:


# see stpacenetutilities.labeltools.corelabeltools createGeoJSONFromRaster


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




