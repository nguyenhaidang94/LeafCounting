import os
import numpy as np
from tensorflow.keras.models import load_model
import traceback
from configs.global_vars import MODEL_DIR, DATA_DIR, IMAGE_SIZE
from dataloader.dataloader import DataLoader
from utils.mail_utils import send_email

def run():
    try:
        print("Load model")
        model_name = os.path.join(MODEL_DIR, 'model_2020-11-08_00-48-42')
        model = load_model(model_name)

        print("Predict")
        img_path = os.path.join(DATA_DIR, 'A1', 'plant001_rgb.png')
        (w,h,c) = IMAGE_SIZE
        x = DataLoader().load_rgb_img(img_path, target_size=(w,h))
        x = np.reshape(x, (1, w, h, c))
        result = model.predict(x)
        print("Number of leaves: {}".format(int(result[0][0])))
        send_email("Prim", "Prediction has finished!")
    except Exception as e:
        traceback.print_exc()
        send_email("Prim error", str(e))

if __name__ == '__main__':
    run()