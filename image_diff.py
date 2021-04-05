# import the necessary packages
import os

from skimage.metrics import structural_similarity, mean_squared_error
import imutils
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def find_box_cords(a):
    r = a.any(1)
    if r.any():
        m, n = a.shape
        c = a.any(0)
        out = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
    else:
        out = (0, 0, 0, 0)
    return out


total_SSIM = 0
total_num_of_frames = 0
total_strips = 0
dir = 'labels/'
labels = list()
for filename in sorted(os.listdir(dir)):
    if filename.endswith('.png'):
        labels.append(cv2.imread(dir + filename))
labels = labels
dir = 'image_strips/TROOP/'
for filename in os.listdir(dir):
    if filename.endswith('.npy'):
        img, flow, bbox, data = np.load(dir + filename, allow_pickle=True)
        total_num_of_frames += len(img)
        old_index = 0
        strip_len = 0
        for Bindex in range(len(img)):
            scores = np.zeros(len(labels))
            scores_MSE = np.zeros(len(labels))
            Originals = []
            Tests = []
            for i, imageA in enumerate(labels):
                imageA = imageA.copy()
                x, x_w, y, y_h = find_box_cords(bbox[Bindex][:, :, 0])
                imageB = img[Bindex][x:x_w, y:y_h]
                ratioA = imageA.shape[1] / imageA.shape[0]  # cv place the width height in revers to numpy
                ratioB = imageB.shape[0] / imageB.shape[1]
                if abs(ratioA - ratioB) > 0.7:
                    scores[i] = -1
                    scores_MSE[i] = 20000
                    Originals.append(imageA)
                    Tests.append(imageB)
                    continue
                interpolation = cv2.INTER_CUBIC if imageA.shape[1] > imageB.shape[0] else cv2.INTER_AREA
                imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation=interpolation)
                grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

                # compute the Structural Similarity Index (SSIM) between the two
                # images, ensuring that the difference image is returned
                (score, diff) = structural_similarity(grayA, grayB, full=True)
                MSE = mean_squared_error(grayA, grayB)
                diff = (diff * 255).astype("uint8")
                # print("SSIM: {}, MSE: {}, lable: {}".format(score, MSE, i))
                scores[i] = score
                scores_MSE[i] = MSE
                # threshold the difference image, followed by finding contours to
                # obtain the regions of the two input images that differ
                thresh = cv2.threshold(diff, 0, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                Originals.append(imageA)
                Tests.append(imageB)
            best_match = int(np.argmax(scores))
            best_match_mse = int(np.argmax(scores_MSE))
            if scores[best_match] > 0.40:
                if old_index + 1 == Bindex:
                    strip_len += 1
                old_index = Bindex
                total_SSIM += 1
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(cv2.cvtColor(Originals[best_match], cv2.COLOR_BGR2RGB))
                axarr[0].set_title('Original')
                axarr[1].imshow(Tests[best_match])
                axarr[1].set_title('Test')
                f.text(0.5, 0.04,
                       f'-SSIM:{scores[best_match]:.5f} MSE:{scores_MSE[best_match]:.2f}\nlable:{best_match}',
                       ha='center', va='center', size='medium')
                # print("-SSIM: {}, MSE: {}, lable: {}".format(scores[best_match], scores_MSE[best_match], best_match))
                plt.show()
            else:
                if strip_len >= 5:
                    total_strips += 1
                strip_len = 0
                old_index = Bindex
print('total', total_SSIM)
print('total_num_of_frames', total_num_of_frames)
print('total_strips', total_strips)
