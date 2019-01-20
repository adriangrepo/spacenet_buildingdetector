#!/usr/bin/env python
# coding: utf-8

# Nadir angle prediction - one of 3 angles

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


import fastai
print(fastai.__version__)


# In[ ]:


torch.cuda.set_device(0)


# In[ ]:


DATE = datetime.datetime.today().strftime('%Y%m%d')
print(f'DATE: {DATE}') 


# In[ ]:


UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}') 


# In[ ]:


ARCH = models.resnet34
ARCH_NAME = 'rn34'
MODEL_NAME = 'pred'


# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# In[ ]:


#using HD here
path = Path('../../data/SpaceNet_Off-Nadir_Dataset')
path.ls()


# In[ ]:


path_img = path/'resized_train_300'
path_lbl = path/'resized_masks_300'
path_test = path/'resized_test_300'


# In[ ]:


path_nadir = path/'by_nadir/train/nadir'
path_off_nadir = path/'by_nadir/train/off_nadir'
path_very_off_nadir = path/'by_nadir/train/very_off_nadir'


# In[ ]:


fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)
fnames[:3], lbl_names[:3] 


# In[ ]:


#subset names based on angle
nadir_fnames=[]
off_nadir_fnames=[]
very_off_nadir_fnames=[]
for aname in fanames:
    apts = str(aname.name).split('_')
    nadir = apts[2]
    angle = int(nadir.split('nadir')[1])
    if angle <= 25:
        nadir_fnames.append(aname)
    elif 26 <= angle <= 40:
        off_nadir_fnames.append(aname)
    elif 41 <= angle <= 55:
        very_off_nadir_fnames.append(aname)


# In[ ]:





# In[ ]:


np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.png$')


# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# ## Training: resnet34

# Now we will start training our model. We will use a [convolutional neural network](http://cs231n.github.io/convolutional-networks/) backbone and a fully connected head with a single hidden layer as a classifier. Don't know what these things mean? Not to worry, we will dive deeper in the coming lessons. For the moment you need to know that we are building a model which will take images as input and will output the predicted probability for each of the categories (in this case, it will have 37 outputs).
# 
# We will train for 4 epochs (4 cycles through all our data).

# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# ## Results

# Let's see what results we have got. 
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly. 
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


doc(interp.plot_top_losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# ## Unfreezing, fine-tuning, and learning rates

# Since our model is working as we expect it to, we will *unfreeze* our model and train some more.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# That's a pretty accurate model!

# ## Training: resnet50

# Now we will train in the same way as before but with one caveat: instead of using resnet34 as our backbone we will use resnet50 (resnet34 is a 34 layer residual network while resnet50 has 50 layers. It will be explained later in the course and you can learn the details in the [resnet paper](https://arxiv.org/pdf/1512.03385.pdf)).
# 
# Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1-50')


# It's astonishing that it's possible to recognize pet breeds so accurately! Let's see if full fine-tuning helps:

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))


# If it doesn't, you can always go back to your previous model.

# In[ ]:


learn.load('stage-1-50');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val=2)


# ## Other data formats

# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE); path


# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit(2)


# In[ ]:


df = pd.read_csv(path/'labels.csv')
df.head()


# In[ ]:


data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))
data.classes


# In[ ]:


data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


fn_paths = [path/name for name in df['name']]; fn_paths[:2]


# In[ ]:


pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes


# In[ ]:


labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]


# In[ ]:


data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes


# In[ ]:




