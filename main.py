import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
import cv2
import numpy as np
import torch.utils.data as utils
from PIL import Image


def make_Video(imgs, name='video'):
    dims = imgs[0].ndim
    if dims > 2:
        height, width, layers = imgs[0].shape
    else:
        height, width = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(name + '.avi', fourcc, 20.0, (width, height))

    for j in range(0, len(imgs)):
        if dims > 2:
            new_image = np.zeros((imgs[j].shape[0], imgs[j].shape[1], 3), dtype=np.int8)
            new_image[:, :, 0] = imgs[j][:, :, 2]
            new_image[:, :, 1] = imgs[j][:, :, 1]
            new_image[:, :, 2] = imgs[j][:, :, 0]
            video.write(new_image)
        else:
            new_image = np.stack((imgs[j],) * 3, axis=-1)
            video.write(new_image)

    video.release()
    cv2.destroyAllWindows()


def chunks(cap, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(cap), n):
        yield utils.Subset(cap, range(i - n, i))


def extractPath(imageId, root, img_type='objects'):
    runId = int(imageId // 10e8)
    mapId = int((imageId - runId * 10e8) // 10e6)
    mapidString = str(mapId)
    if (mapId < 10):
        mapidString = '0' + mapidString
    tic = int(imageId - runId * 10e8 - mapId * 10e6)
    string_tic = str(tic)
    zeros = ''
    for i in range(6 - len(string_tic)):
        zeros = '0' + zeros
    return root + '/run' + str(runId) + '/map' + mapidString + '/' + img_type + '/' + zeros + str(tic) + '.png'


def chunkToObjectsImages(chunk, root, img_type='objects'):
    object_image_paths = []
    prev_id = 0
    for img, target in chunk:
        imageId = target[0]['image_id'] if len(target) > 0 else prev_id + 1
        prev_id = imageId
        if imageId is not None:
            object_image_paths.append(extractPath(imageId, root, img_type=img_type))
        else:
            object_image_paths.append(None)
    images = map(lambda path: np.zeros((200, 320, 3), dtype=np.int8) if path is None else
    np.array(Image.open(path) if img_type != 'depth' else cv2.imread(path, cv2.IMREAD_GRAYSCALE), dtype=np.int8),
                 object_image_paths)
    return list(images)


def main(time_tic=10 * 35):
    cap = dset.CocoDetection(root='../cocodoom',
                             annFile='../cocodoom/run-full-test.json')

    print('Number of samples: ', len(cap))

    subsets = list(chunks(cap, time_tic))
    set_num = 50
    depth_vid = chunkToObjectsImages(list(subsets[set_num]), cap.root, img_type='depth')
    object_vid = chunkToObjectsImages(list(subsets[set_num]), cap.root)
    vid = list(map(lambda x: np.array(x[0]), list(subsets[set_num])))
    make_Video(np.array(vid), 'rgb')
    make_Video(np.array(object_vid), 'object')
    make_Video(np.array(depth_vid), 'depth')


if __name__ == '__main__':
    main(20 * 35)
