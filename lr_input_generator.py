from glob import glob
from pathlib import Path
import cv2
import os


SCALE = 3

def modcrop(image, scale=3):
  (h, w) = image.shape[:2]
  w -= int(w % scale)
  h -= int(h % scale)
  image = image[0 : h, 0 : w]
  return image

def main():

    Path('400X/lr_predict').mkdir()
    Path('400X/lr_predict/benign').mkdir()
    Path('400X/lr_predict/malignant').mkdir()

    predict_400X_benign= glob('400X/predict/benign/*.png')
    for image_path in predict_400X_benign:
        print(image_path)
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = modcrop(image,SCALE)
        shape = image.shape
        lr_image = cv2.resize(image, ((int)(shape[1] / SCALE), (int)(shape[0] / SCALE)), cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, ((int)(shape[1]), (int)(shape[0])), cv2.INTER_CUBIC)
        new_file_path = '400X/lr_predict/benign/' + os.path.basename(image_path)
        cv2.imwrite(new_file_path, lr_image)

    predict_400X_malignant= glob('400X/predict/malignant/*.png')
    for image_path in predict_400X_malignant:
        print(image_path)
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = modcrop(image,SCALE)
        shape = image.shape
        lr_image = cv2.resize(image, ((int)(shape[1] / SCALE), (int)(shape[0] / SCALE)), cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, ((int)(shape[1]), (int)(shape[0])), cv2.INTER_CUBIC)
        new_file_path = '400X/lr_predict/malignant/' + os.path.basename(image_path)
        cv2.imwrite(new_file_path, lr_image)


if __name__ == '__main__':
    main()
