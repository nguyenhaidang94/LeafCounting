import os
import glob
import cv2
import re
import pandas as pd
import numpy as np
from utils.data_utils import train_val_test_split

class DataLoader(object):

    def _load_rgb_img(self, img_path, target_size=None):
        bgr_img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        if target_size != None:
            bgr_img = cv2.resize(bgr_img, target_size)
        rgb_img = bgr_img[...,::-1]/255.0
        return rgb_img

    def _load_data_in_subdir(self, base_dir, sub_dir, target_size):
        img_pattern_path = os.path.join(base_dir, sub_dir, 'plant*_rgb.png')
        img_paths = glob.glob(img_pattern_path)
        # read csv containing leaf counting
        csv_filename = os.path.join(base_dir, sub_dir, sub_dir+'.csv')
        leaf_count_csv = pd.read_csv(csv_filename, header=None, names=['name', 'count'])
        # read each image and corresponding label
        file_regex = FILE_REGEX_TEMPLATE.format(sub_dir)
        X = []
        y = []
        for img_path in img_paths:
            match_result = re.search(file_regex, img_path)
            if match_result:
                name = match_result.group(1)
                values = leaf_count_csv.loc[leaf_count_csv['name']==name, 'count'].values
                if len(values) > 0:
                    # get image, label
                    X.append(self._load_rgb_img(img_path, target_size))
                    y.append(values[0])
                else:
                    print("Couldn't find count with name {} in the csv file".format(name))
            else:
                print("Couldn't extract file name from path: {}".format(img_path))
        return X, y


    def load_data(self, base_dir, sub_dirs, target_size, train_ratio, val_ratio):
        (w, h, c) = target_size
        X_train_full = []
        y_train_full = []
        X_val_full = []
        y_val_full = []
        X_test_full = []
        y_test_full = []
        for sub_dir in sub_dirs:
            X, y = self._load_data_in_subdir(base_dir, sub_dir, target_size=(w,h))
            X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_ratio=train_ratio, val_ratio=val_ratio)
            X_train_full = X_train_full + X_train
            y_train_full = y_train_full + y_train
            X_val_full = X_val_full + X_val
            y_val_full = y_val_full + y_val
            X_test_full = X_test_full + X_test
            y_test_full = y_test_full + y_test  
        X_train_full = np.reshape(X_train_full, (len(X_train_full), w, h, c) )
        X_val_full = np.reshape(X_val_full, (len(X_val_full), w, h, c) )
        X_test_full = np.reshape(X_test_full, (len(X_test_full), w, h, c) )

        return X_train_full, y_train_full, X_val_full, y_val_full, X_test_full, y_test_full