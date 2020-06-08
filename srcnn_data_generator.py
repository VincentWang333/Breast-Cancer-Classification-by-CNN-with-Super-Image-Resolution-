from glob import glob
import os
from pathlib import Path
import shutil
import random
TRAIN = 6
VALIDATE = 2




def duplicate(num,list):
    if len(list)==0:
        return False
    for i in list:
        if i == num:
            return True
    return False

def main():

    Path('SRCNN').mkdir()
    Path('SRCNN/400X').mkdir()
    Path('SRCNN/400X/train').mkdir()
    Path('SRCNN/400X/validate').mkdir()




    benign_types = glob('BreakHis/benign/SOB/*')
    for benign_type in benign_types:
        benign_patients = glob(benign_type+'/*')
        for benign_patient in benign_patients:
            print(benign_patient)

            benign_patient_400X = glob(benign_patient+'/400X/*.png')
            benign_patient_400X_used = []
            for i in range(TRAIN):
                index = random.randrange(len(benign_patient_400X))
                while(duplicate(index,benign_patient_400X_used)):
                    index = random.randrange(len(benign_patient_400X))
                image = benign_patient_400X[index]
                shutil.copy2(image, 'SRCNN/400X/train')
            for i in range(VALIDATE):
                index = random.randrange(len(benign_patient_400X))
                while(duplicate(index,benign_patient_400X_used)):
                    index = random.randrange(len(benign_patient_400X))
                image = benign_patient_400X[index]
                shutil.copy2(image, 'SRCNN/400X/validate')


    malignant_types = glob('BreakHis/malignant/SOB/*')
    for malignant_type in malignant_types:
        maignant_patients = glob(malignant_type+'/*')
        for malignant_patient in maignant_patients:
            print(malignant_patient)

            malignant_patient_400X = glob(malignant_patient+'/400X/*.png')
            malignant_patient_400X_used = []
            for i in range(TRAIN):
                index = random.randrange(len(malignant_patient_400X))
                while(duplicate(index,malignant_patient_400X_used)):
                    index = random.randrange(len(malignant_patient_400X))
                image = malignant_patient_400X[index]
                shutil.copy2(image, 'SRCNN/400X/train')
            for i in range(VALIDATE):
                index = random.randrange(len(malignant_patient_400X))
                while(duplicate(index,malignant_patient_400X_used)):
                    index = random.randrange(len(malignant_patient_400X))
                image = malignant_patient_400X[index]
                shutil.copy2(image, 'SRCNN/400X/validate')









if __name__ == '__main__':
    main()
