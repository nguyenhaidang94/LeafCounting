import os
import numpy as np
from tensorflow.keras.models import load_model
from configs.global_vars import MODEL_DIR, DATA_DIR, IMAGE_SIZE
from dataloader.dataloader import DataLoader

def run():
    print("Load model")
    model_name = os.path.join(MODEL_DIR, 'model_2020-11-08_00-48-42')
    model = load_model(model_name)

    print("Predict")
    img_path = os.path.join(DATA_DIR, 'A1', 'plant001_rgb.png')
    (w,h,c) = IMAGE_SIZE
    x = DataLoader().load_rgb_img(img_path, target_size=(w,h))
    x = np.reshape(x, (1, w, h, c))
    result = model.predict(x)
    print(result)
    print(int(result[0][0]))

if __name__ == '__main__':
    run()