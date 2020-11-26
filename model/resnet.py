from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from model.base_model import BaseModel

class Resnet(BaseModel):
    
    def __init__(self):
        super().__init__()

    def build(self):
        base = ResNet50(include_top=False, input_shape=self.image_size)
        x = Flatten()(base.output)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        self.model = Model(inputs=base.inputs, outputs=x)