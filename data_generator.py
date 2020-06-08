from glob import glob
from pathlib import Path
import shutil
import os
import random


Path('400X').mkdir()
Path('400X/train').mkdir()
Path('400X/validate').mkdir()
Path('400X/predict').mkdir()
Path('400X/train/malignant').mkdir()
Path('400X/validate/malignant').mkdir()
Path('400X/predict/malignant').mkdir()
Path('400X/train/benign').mkdir()
Path('400X/validate/benign').mkdir()
Path('400X/predict/benign').mkdir()


def duplicate(num,list):
    if len(list)==0:
        return False
    for i in list:
        if i == num:
            return True
    return False

benign_400X_types = glob('BreakHis/benign/SOB/*')
for benign_400X_type in benign_400X_types:
    print(benign_400X_type)
    benign_400X_type_patients = glob(benign_400X_type+'/*')
    num_patients = len(benign_400X_type_patients)
    patients_used = []
    for i in range(16):
        index = random.randrange(num_patients)
        while(duplicate(index,patients_used)):
            index = random.randrange(num_patients)
        patient = benign_400X_type_patients[index]
        patient_400x_images = glob(patient+'/400X/*.png')
        for image in patient_400x_images:
            shutil.copy2(image, '400X/train/benign')
    for i in range(4):
        index = random.randrange(num_patients)
        while(duplicate(index,patients_used)):
            index = random.randrange(num_patients)
        patient = benign_400X_type_patients[index]
        patient_400x_images = glob(patient+'/400X/*.png')
        for image in patient_400x_images:
            shutil.copy2(image, '400X/validate/benign')
    for i in range(4):
        index = random.randrange(num_patients)
        while(duplicate(index,patients_used)):
            index = random.randrange(num_patients)
        patient = benign_400X_type_patients[index]
        patient_400x_images = glob(patient+'/400X/*.png')
        for image in patient_400x_images:
            shutil.copy2(image, '400X/predict/benign')





malignant_400X_types = glob('BreakHis/malignant/SOB/*')
for malignant_400X_type in malignant_400X_types:
    print(malignant_400X_type)
    malignant_400X_type_patients = glob(malignant_400X_type+'/*')
    num_patients = len(malignant_400X_type_patients)
    patients_used = []
    for i in range(12):
        index = random.randrange(num_patients)
        while(duplicate(index,patients_used)):
            index = random.randrange(num_patients)
        patient = malignant_400X_type_patients[index]
        patient_400x_images = glob(patient+'/400X/*.png')
        for image in patient_400x_images:
            shutil.copy2(image, '400X/train/malignant')
    for i in range(3):
        index = random.randrange(num_patients)
        while(duplicate(index,patients_used)):
            index = random.randrange(num_patients)
        patient = malignant_400X_type_patients[index]
        patient_400x_images = glob(patient+'/400X/*.png')
        for image in patient_400x_images:
            shutil.copy2(image, '400X/validate/malignant')
    for i in range(38):
        index = random.randrange(num_patients)
        while(duplicate(index,patients_used)):
            index = random.randrange(num_patients)
        patient = malignant_400X_type_patients[index]
        patient_400x_images = glob(patient+'/400X/*.png')
        for image in patient_400x_images:
            shutil.copy2(image, '400X/predict/malignant')
