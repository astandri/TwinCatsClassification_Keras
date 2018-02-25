import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#================ Functions ====================
def load_myCat_as_dataframe(catname):
    foldername = (catname)
    filenames = glob.glob('{}/*.jpg'.format(foldername))
    
    images = []

    for filename in filenames:
        img = cv2.imread(filename,0)

        #reshape image
        r = 100.0 / img.shape[1]
        dim = (100, int(img.shape[0] * r))
        resized_img = cv2.resize(img, dim).reshape(1,-1)
        images.append(pd.DataFrame(resized_img))
    
    return pd.concat(images).reset_index(drop=True)

def split_dataset(df):
    #shuffle the index
    shuffled_df = df.sample(frac=1)
    
    #split dataset to train, validation, test with 70:20:10 ratio
    samples_count = shuffled_df.shape[0]
    train_samples_count = int(0.8*samples_count)
    validation_samples_count = int(0.1*samples_count)
    
    train_df = shuffled_df.iloc[:train_samples_count]
    validation_df = shuffled_df.iloc[train_samples_count:train_samples_count+validation_samples_count]
    test_df = shuffled_df.iloc[train_samples_count+validation_samples_count:]
    
    return (train_df, validation_df, test_df)

#==============================================

claudia = load_myCat_as_dataframe('Images/Claudia')
claudia['label'] = 0

lucy = load_myCat_as_dataframe('Images/Lucy')
lucy['label'] = 1

print('Done Loading Images')

#split dataset
train_claudia, validation_claudia, test_claudia = split_dataset(claudia)
train_lucy, validation_lucy, test_lucy = split_dataset(lucy)

print('Done Splitting to Train, Validation and Test')

#join dataset
train = pd.concat([train_claudia, train_lucy]).sample(frac=1).reset_index(drop=True)
validation = pd.concat([validation_claudia, validation_lucy]).sample(frac=1).reset_index(drop=True)
test = pd.concat([test_claudia, test_lucy]).sample(frac=1).reset_index(drop=True)

#Save to csv
train.to_csv('Dataset/train.csv', index=False)
validation.to_csv('Dataset/validation.csv', index=False)
test.to_csv('Dataset/test.csv', index=False)

print('Dataset saved to as .csv')
