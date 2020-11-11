import time
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from dataloader.dataloader import DataLoader
from utils.time_utils import get_current_time
from configs.global_vars import MODEL_DIR
from configs.global_vars import ROTATION_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP

class Resnet(object):
    
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.image_size = None

    def _preprocess_data(self, batch_size):
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow(self.X_test, self.y_test, batch_size=batch_size)
        return train_generator, val_generator, test_generator

    def load_data(self, base_dir, sub_dirs, image_size, train_ratio, val_ratio):
        self.image_size = image_size
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test \
            = DataLoader().load_data(base_dir, sub_dirs, image_size, train_ratio, val_ratio)
        print("X_train:", self.X_train.shape)
        print("X_val:", self.X_val.shape)
        print("X_test:", self.X_test.shape)

    def build(self, train_base_model=True):
        base = ResNet50(include_top=False, input_shape=self.image_size)
        if train_base_model == False:
            base.trainable = False
        x = Flatten()(base.output)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        self.model = Model(inputs=base.inputs, outputs=x)

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
            callback = EarlyStopping( monitor='val_loss', patience=ES_EPOCHS, mode='min', restore_best_weights=True)
            history = self.model.fit(train_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch\
                , validation_data=val_generator, callbacks=[callback])
        else:
            history = self.model.fit(train_generator, epochs=n_epochs, steps_per_epoch=steps_per_epoch\
                , validation_data=val_generator)
        return history

    def evaluate(self):
        y_test_np = np.array(self.y_test)
        return self.model.evaluate(self.X_test, y_test_np)

    def save(self):
        model_folder = os.path.join(MODEL_DIR, "model_{}".format(get_current_time()))
        self.model.save(model_folder)

    def predict(self, img_path):
        (w,h,c) = self.image_size
        x = DataLoader().load_rgb_img(img_path, target_size=(w,h))
        x = np.reshape(x, (1, w, h, c))
        result = self.model.predict(x)
        return int(result[0][0])