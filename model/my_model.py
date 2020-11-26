from tensorflow import keras
from tensorflow.keras import layers
from model.base_model import BaseModel

class MyModel(BaseModel):

    def __init__(self):
        super().__init__()

    def build(self):
        inputs = keras.Input(shape=self.image_size)
        x = layers.Conv2D(32, (3,3), activation="relu", padding="same",)(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = layers.Conv2D(64, (3,3), activation="relu", padding="same",)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = layers.Conv2D(128, (3,3), activation="relu", padding="same",)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = layers.Conv2D(256, (3,3), activation="relu", padding="same",)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = layers.Conv2D(512, (3,3), activation="relu", padding="same",)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1, activation='linear')(x)
        self.model = keras.Model(inputs=inputs, outputs=x)