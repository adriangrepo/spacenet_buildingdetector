import PIL.Image as pil_image
import os
import glob
from fastai import *
from fastai.vision import *


def crop(im, height, width):
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)

def crop_h(im, height, width):
    imgwidth, imgheight = im.size
    for i in range((imgheight // height)):
        for j in range((imgwidth // width)-1):
            box = ((j * width)+int(width/2), i * height, ((j + 1) * width)+int(width/2), (i + 1) * height)
            #print(f'h box: {box}')
            yield im.crop(box)

def crop_v(im, height, width):
    imgwidth, imgheight = im.size
    for i in range((imgheight // height)-1):
        for j in range((imgwidth // width)):
            box = (j * width, (i * height)+int(height/2), (j + 1) * width, ((i + 1) * height)+int(height/2))
            #print(f'v box: {box}')
            yield im.crop(box)

def crop_c(im, height, width):
    imgwidth, imgheight = im.size
    for i in range((imgheight // height)-1):
        for j in range((imgwidth // width)-1):
            box = ((j * width)+int(width/2), (i * height)+int(height/2), ((j + 1) * width)+int(width/2), ((i + 1) * height)+int(height/2))
            #print(f'c box: {box}')
            yield im.crop(box)

def workflow(imgdir, outdir):
    basename = '*.tif'
    filelist = get_image_files(imgdir)
    for filename in filelist:
        im = pil_image.open(filename)
        imgwidth, imgheight = im.size
        height = int(imgheight / 3)
        width = int(imgwidth / 3)
        start_num = 0
        for k, piece in enumerate(crop(im, height, width), start_num):
            img = pil_image.new('RGB', (width, height), 255)
            img.paste(piece)
            img.save(outdir / f'{filename.stem}_g_{k}.png')
        '''
        #just create a 3x3 grid, and recombine cut pieces - much simpler and significantly less data (9x vs 25x)
        for k, piece in enumerate(crop_h(im, height, width), start_num):
            img = pil_image.new('RGB', (width, height), 255)
            img.paste(piece)
            img.save(outdir / f'{filename.stem}_h_{k}.png')
        for k, piece in enumerate(crop_v(im, height, width), start_num):
            img = pil_image.new('RGB', (width, height), 255)
            img.paste(piece)
            img.save(outdir / f'{filename.stem}_v_{k}.png')
        for k, piece in enumerate(crop_c(im, height, width), start_num):
            img = pil_image.new('RGB', (width, height), 255)
            img.paste(piece)
            img.save(outdir / f'{filename.stem}_c_{k}.png')
        '''

if __name__ == '__main__':
    imgdir = Path('../../../data/SpaceNet_Off-Nadir_Dataset/train_rgb')
    outdir = Path('../../../data/SpaceNet_Off-Nadir_Dataset/resized_train_300')
    workflow(imgdir, outdir)
    imgdir = Path('../../../data/SpaceNet_Off-Nadir_Dataset/masks')
    outdir = Path('../../../data/SpaceNet_Off-Nadir_Dataset/resized_masks_300')
    workflow(imgdir, outdir)
    imgdir = Path('../../../data/SpaceNet_Off-Nadir_Dataset/test_rgb')
    outdir = Path('../../../data/SpaceNet_Off-Nadir_Dataset/resized_test_300')
    workflow(imgdir, outdir)


