from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2 as cv
from PIL import Image, ImageFilter

def calc_optical_flow(video):
    """
    :param video: video
    :type video:np.array
    :return: optical flow as an array of len(video)-1
    :rtype: np.array
    """
    img_array_new = np.zeros_like(video, dtype=np.uint8)
    size = (video.shape[1], video.shape[2])
    for i in range(len(video)):
        new_image = np.zeros((video[i].shape[0], video[i].shape[1], 3), dtype=np.uint8)
        new_image[:, :, 0] = video[i][:, :, 2]
        new_image[:, :, 1] = video[i][:, :, 1]
        new_image[:, :, 2] = video[i][:, :, 0]
        img_array_new[i] = new_image

    first_frame = img_array_new[0]

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255
    out_optical = np.zeros((len(img_array_new) - 1, *size, 3), dtype=np.uint8)
    for i in range(1, len(img_array_new)):
        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        frame = img_array_new[i]
        # Opens a new window and displays the input
        # frame
        # cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # from shape (channel , width , height) to (width , height , channel)
        # rgb = rgb.transpose((2, 1, 0))
        # Flip image to back to rgb
        flip_rgb = np.zeros_like(rgb, dtype=np.uint8)
        flip_rgb[:, :, 0] = rgb[:, :, 2]
        flip_rgb[:, :, 1] = rgb[:, :, 1]
        flip_rgb[:, :, 2] = rgb[:, :, 0]

        out_optical[i - 1] = flip_rgb

        # Updates previous frame
        prev_gray = gray
    return out_optical


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


def get_Masked_Strips(filterClasses: list, coco, dataDir, section):
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    catIds = coco.getCatIds(catNms=filterClasses)
    print(catIds)
    imgIds = set()
    for id in catIds:
        imgIds |= set(coco.getImgIds(catIds=[id]))
    imgIds = list(imgIds)
    print("Number of images containing all the  classes:", len(imgIds))
    strips_ids = find_sequences_in_list(imgIds, seq_len=11)
    print("Number of continous images containing all the classes:", len(strips_ids))
    strips_list = []
    for i in range(section[0] * (len(strips_ids) // 50),
                   section[1] * (len(strips_ids) // 50)):  # dividing for limiting the amount
        strips_list.append(get_Mask_Strip(filterClasses, coco, dataDir, cats, catIds, imgIds, strips_ids[i]))
    return strips_list


dataDir = '../cocodoom'
dataType = 'train'
annFile = '{}/run-full-{}.json'.format(dataDir, dataType)

# Initialize the COCO api for instance annotations
coco = COCO(annFile)

# print('The class name is', getClassName(77, cats))


filterClasses = ['TROOP']
# filterClasses = ['TROOP', 'SHOTGUY', 'CHAINGUY', 'UNDEAD', 'HEAD', 'POSSESSED']

#### GENERATE A SEGMENTATION MASK ####
# filterClasses = ['TROOP', 'POSSESSED', 'SHOTGUY', 'HEAD', 'FIRE', 'CHAINGUY', 'MISC2', 'UNDEAD', 'TRACER', 'MISC19',
#                  'MISC43', 'HEADSHOT', 'TFOG', 'SKULL', 'BRUISER', 'BLOOD', 'SERGEANT']
for classes in filterClasses:
    os.makedirs('image_strips/' + classes)
    print('cutting', classes)
    strip_index = 1
    for k in range(2):
        print('section', k, 'of', 50)
        strips = get_Masked_Strips([classes], coco, dataDir, (k, k + 1))
        item_key_to_seq_masked = dict()
        for strip in strips:
            for frame in strip:
                for item_id, mask in frame:
                    if int(item_id % 1e6 + strip_index * 1e6) in item_key_to_seq_masked.keys():
                        item_key_to_seq_masked[int(item_id % 1e6 + strip_index * 1e6)].append(mask)
                    else:
                        item_key_to_seq_masked[int(item_id % 1e6 + strip_index * 1e6)] = [mask]
            strip_index += 1
        item_key_to_seq_masked = {key: value for (key, value) in item_key_to_seq_masked.items() if len(value) == 11}
        amount_to_save = len(item_key_to_seq_masked.keys())
        print(amount_to_save, 'amount to save')
        for step, key in enumerate(item_key_to_seq_masked.keys()):
            # print(key)
            if len(item_key_to_seq_masked[key]) == 11:
                # print(len(item_key_to_seq_masked[key]))
                # print(item_key_to_seq_masked[key])
                if step % 10 == 0:
                    print(step, 'out of ', amount_to_save)
                imgs = item_key_to_seq_masked[key]
                for i in range(len(imgs)):
                    img = Image.fromarray(np.uint8(imgs[i] * 255), "RGB")
                    new_img = img.resize((512, 256))
                    imgs[i] = np.array(new_img)
                imgs = np.array(imgs)
                optical_flow = calc_optical_flow(imgs)
                # print(len(optical_flow) , 'optical flow done')
                np.save('./image_strips/' + classes + '/' + str(key).zfill(8) + '.npy', imgs)
                np.save('./image_strips/' + classes + '/' + str(key).zfill(8) + '-of' +'.npy', optical_flow)
