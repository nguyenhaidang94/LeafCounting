import time
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

from dataloader.dataloader import DataLoader
from utils.time_utils import get_current_time
from utils import eval_utils
from configs.global_vars import MODEL_DIR, RESULT_DIR
from configs.global_vars import ROTATION_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP

class BaseModel(object):
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.image_size = None

    def _predict(self, X):
        y_predict = self.model.predict(X)
        return np.ravel(y_predict)

    def load_data(self, base_dir, sub_dirs, image_size, train_ratio, val_ratio):
        self.image_size = image_size
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test \
            = DataLoader().load_data(base_dir, sub_dirs, image_size, train_ratio, val_ratio)
        print("X_train:", self.X_train.shape)
        print("X_val:", self.X_val.shape)
        print("X_test:", self.X_test.shape)

    def build(self):
        return None

    def compile(self, optimizer):
        self.model.compile(loss='mse', optimizer=optimizer)

    def summary(self):
        self.model.summary()

    def train(self, batch_size, n_epochs, early_stopping_epochs=None):
        train_datagen = ImageDataGenerator(rotation_range=ROTATION_RANGE, zoom_range=ZOOM_RANGE\
            , horizontal_flip=HORIZONTAL_FLIP, vertical_flip=VERTICAL_FLIP)
        train_generator = train_datagen.flow(self.X_train, self.y_train, batch_size=batch_size)
        val_datagen = ImageDataGenerator()
        val_generator = val_datagen.flow(self.X_val, self.y_val, batch_size=batch_size)
        steps_per_epoch = int(self.X_train.shape[0]/batch_size)
        if early_stopping_epochs != None:
            callback = EarlyStopping( monitor='val_loss', patience=early_stopping_epochs, mode='min', restore_best_weights=True)
            history = self.model.fit(train_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch\
                , validation_data=val_generator, callbacks=[callback])
        else:
            history = self.model.fit(train_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch\
                , validation_data=val_generator)
        return history

    def loss_in_test_set(self):
        y_predict = self._predict(self.X_test)
        return eval_utils.mse(self.y_test, y_predict)

    def evaluate(self):
        y_predict = self._predict(self.X_test)
        mse = eval_utils.mse(self.y_test, y_predict)
        mae = eval_utils.mae(self.y_test, y_predict)
        y_predict_rounded = np.rint(y_predict)
        adic = eval_utils.mae(self.y_test, y_predict_rounded)
        acc = eval_utils.acc(self.y_test, y_predict_rounded)
        acc_diff1 = eval_utils.acc_diff1(self.y_test, y_predict_rounded)
        acc_diff2 = eval_utils.acc_diff2(self.y_test, y_predict_rounded)
        print("Mean squared error:", mse)
        print("Mean absolute error:", mae)
        print("Mean absolute difference in count:", adic)
        print("Accuracy: {}%".format(acc*100))
        print("Accuracy +-1: {}%".format(acc_diff1*100))
        print("Accuracy +-2: {}%".format(acc_diff2*100))

        df = pd.DataFrame()
        df['y_true'] = self.y_test
        df['y_predict'] = y_predict
        df['y_predict_rounded'] = y_predict_rounded
        result_file = os.path.join(RESULT_DIR, "result_{}.csv".format(get_current_time()))
        print("Save result to file: {}".format(result_file))
        df.to_csv(result_file, index=False)

    def save(self):
        model_folder = os.path.join(MODEL_DIR, "model_{}".format(get_current_time()))
        print("Save model to folder: {}".format(model_folder))
        self.model.save(model_folder)

    def predict(self, img_path):
        (w,h,c) = self.image_size
        x = DataLoader().load_rgb_img(img_path, target_size=(w,h))
        x = np.reshape(x, (1, w, h, c))
        result = self.model.predict(x)
        return int(result[0][0])