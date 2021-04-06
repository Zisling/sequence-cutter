import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity, mean_squared_error


def find_box_cords(a):
    r = a.any(1)
    if r.any():
        m, n = a.shape
        c = a.any(0)
        out = (r.argmax(), m - r[::-1].argmax(), c.argmax(), n - c[::-1].argmax())
    else:
        out = (0, 0, 0, 0)
    return out


if __name__ == '__main__':
    im_count = 0
    total = 0
    strip_num = 0
    dirr = 'image_strips/TROOP/'
    for filename in os.listdir(dirr):
        if filename.endswith('.npy'):
            img, flow, bbox, data = np.load(dirr + filename, allow_pickle=True)
            total += len(img)
            strip_len = 0
            jumped_frame = False

            for i in range(len(img) - 1):
                idxA = i
                idxB = i + 1
                if jumped_frame:
                    idxA = i - 1
                xA, x_wA, yA, y_hA = find_box_cords(bbox[idxA][:, :, 0])
                xB, x_wB, yB, y_hB = find_box_cords(bbox[idxB][:, :, 0])
                imageA = img[idxA][xA:x_wA, yA:y_hA]
                imageB = img[idxB][xB:x_wB, yB:y_hB]

                ratioA = imageA.shape[1] / imageA.shape[0]
                ratioB = imageB.shape[1] / imageB.shape[0]
                if ratioA < 0.5 or ratioA > 3.5 or ratioB < 0.5 or ratioB > 3.5:
                    strip_len = 0
                    jumped_frame = False
                    continue

                interpolation = cv2.INTER_CUBIC if imageA.shape[0] > imageB.shape[0] else cv2.INTER_AREA
                imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation=interpolation)
                grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
                grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

                # compute the Structural Similarity Index (SSIM) between the two
                # images, ensuring that the difference image is returned
                try:
                    MSE = mean_squared_error(grayA, grayB)
                except ValueError:
                    print(f'Failed on MSE')
                    strip_len = 0
                    jumped_frame = False
                    continue
                if strip_len == 11:
                    strip_num += 1
                    strip_len = 0
                    jumped_frame = False

                if MSE < 200:
                    strip_len += 1
                    im_count += 1
                    jumped_frame = False
                    # f, axarr = plt.subplots(1, 2)
                    # axarr[0].imshow(imageA)
                    # axarr[0].set_title('A for ass')
                    # axarr[1].imshow(imageB)
                    # axarr[1].set_title('B for butt')
                    # f.text(0.5, 0.04,
                    #        f'-MSE:{MSE:.2f}\nratio A:{ratioA} ratio B:{ratioB}',
                    #        ha='center', va='center', size='medium')
                    # plt.show()
                else:
                    if jumped_frame:
                        strip_len = 0
                        jumped_frame = False
                    else:
                        jumped_frame = True

    print(f'total: {total} im_count:{im_count} strip_num:{strip_num}')
