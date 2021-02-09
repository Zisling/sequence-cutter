from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


def crate_Mask(anns, filterClasses, coco, img, cats):
    #### GENERATE A SEGMENTATION MASK ####
    mask_list = []
    mask_to_id = []
    for i in range(len(anns)):
        mask_res = np.zeros((img['height'], img['width']))
        className = getClassName(anns[i]['category_id'], cats)
        if className in filterClasses:
            mask_res = np.maximum(coco.annToMask(anns[i]), mask_res)
            mask_list.append(mask_res)
            mask_to_id.append(anns[i]['id'])
    return mask_list, mask_to_id


def get_Mask(filterClasses: list, coco, dataDir, cats, catIds, imgIds, image_id=None):
    # Load the categories in a variable
    if image_id is None:
        # load and display a random image
        image_id = np.random.randint(0, len(imgIds))
    img = coco.loadImgs(imgIds[image_id])[0]  # remove from list
    I = io.imread('{}/{}'.format(dataDir, img['file_name'])) / 255.0  # load real image
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    masks, items_id = crate_Mask(anns, filterClasses, coco, img, cats)
    return [I[:, :, 0:3] * np.reshape(Mask, (200, 320, 1)) for Mask in masks], items_id


def find_sequences_in_list(img_ids, seq_len=10):
    old_id = img_ids[0]
    seq = [0]
    index = 1
    seq_list = []
    for ID in img_ids[1:]:
        if len(seq) == seq_len:
            seq_list.append(seq)
            seq = [index]
        elif ID - 1 == old_id:
            seq.append(index)
        else:
            seq = [index]
        old_id = ID
        index += 1
    return seq_list


def get_Mask_Strip(filterClasses: list, coco, dataDir, cats, catIds, imgIds, strips_ids):
    strip_and_item_id = []
    for ID in strips_ids:
        masks, items_id = get_Mask(filterClasses, coco, dataDir, cats, catIds, imgIds, image_id=ID)
        strip_and_item_id.append(zip(items_id, masks))
    return strip_and_item_id


def get_Masked_Strips(filterClasses: list, coco, dataDir):
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)

    catIds = coco.getCatIds(catNms=filterClasses)
    imgIds = coco.getImgIds(catIds=catIds)
    print("Number of images containing all the  classes:", len(imgIds))
    strips_ids = find_sequences_in_list(imgIds, seq_len=16)
    strips_list = []
    for i in range(len(strips_ids) // 100):  # dividing for limiting the amount
        strips_list.append(get_Mask_Strip(filterClasses, coco, dataDir, cats, catIds, imgIds, strips_ids[i]))
    return strips_list


dataDir = '../cocodoom'
dataType = 'train'
annFile = '{}/run-full-{}.json'.format(dataDir, dataType)

# Initialize the COCO api for instance annotations
coco = COCO(annFile)

# print('The class name is', getClassName(77, cats))


filterClasses = ['TROOP']

#### GENERATE A SEGMENTATION MASK ####
# filterClasses = ['TROOP', 'POSSESSED', 'SHOTGUY', 'HEAD', 'FIRE', 'CHAINGUY', 'MISC2', 'UNDEAD', 'TRACER', 'MISC19',
#                  'MISC43', 'HEADSHOT', 'TFOG', 'SKULL', 'BRUISER', 'BLOOD', 'SERGEANT']

strips = get_Masked_Strips(filterClasses, coco, dataDir)
item_key_to_seq_masked = dict()
strip_index = 1
for strip in strips:
    for frame in strip:
        for item_id, mask in frame:
            if int(item_id % 1e6 + strip_index * 1e6) in item_key_to_seq_masked.keys():
                item_key_to_seq_masked[int(item_id % 1e6 + strip_index * 1e6)].append(mask)
            else:
                item_key_to_seq_masked[int(item_id % 1e6 + strip_index * 1e6)] = [mask]
    strip_index += 1
for key in item_key_to_seq_masked.keys():
    print(key)
    if len(item_key_to_seq_masked[key]) >= 10:
        print(len(item_key_to_seq_masked[key]))
        for mask in item_key_to_seq_masked[key]:
            plt.imshow(mask)
            plt.show()
