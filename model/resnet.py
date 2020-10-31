from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataloader.dataloader import DataLoader

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

    def load_data(self, base_dir, sub_dirs, image_size, train_ratio, val_ratio):
        self.image_size = image_size
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test \
            = DataLoader().load_data(base_dir, sub_dirs, image_size, train_ratio, val_ratio)

    def build(self):
        base = ResNet50(include_top=False, input_shape=self.image_size)
        x = Flatten()(base.output)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        self.model = Model(inputs=base.inputs, outputs=x)

    def compile(self, optimizer):
        self.model.compile(loss='mse', optimizer=optimizer)

    def summary(self):
        print(self.model.summary())