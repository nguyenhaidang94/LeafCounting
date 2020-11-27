import os
import numpy as np
import traceback
from configs.global_vars import MODEL_DIR, DATA_DIR, IMAGE_SIZE
from model.my_model import MyModel
from dataloader.dataloader import DataLoader
from utils.mail_utils import send_email

WEIGHTS_FILE = ''

def run():
    try:
        print("Create model")
        model = MyModel()
        model.image_size = IMAGE_SIZE
        print("Build model")
        model.build()
        print("Load weights")
        weights_file = os.path.join(MODEL_DIR, WEIGHTS_FILE)
        model.load_weights(weights_file)
        print("Load samples")
        img_paths = []
        img_paths.append(os.path.join(DATA_DIR, 'A1', 'plant001_rgb.png'))
        img_paths.append(os.path.join(DATA_DIR, 'A1', 'plant161_rgb.png'))
        img_paths.append(os.path.join(DATA_DIR, 'A2', 'plant001_rgb.png'))
        img_paths.append(os.path.join(DATA_DIR, 'A2', 'plant039_rgb.png'))
        img_paths.append(os.path.join(DATA_DIR, 'A3', 'plant001_rgb.png'))
        img_paths.append(os.path.join(DATA_DIR, 'A3', 'plant027_rgb.png'))
        img_paths.append(os.path.join(DATA_DIR, 'A4', 'plant0001_rgb.png'))
        img_paths.append(os.path.join(DATA_DIR, 'A4', 'plant0857_rgb.png'))
        (w,h,c) = IMAGE_SIZE
        print("Predict")
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            print("Image: {}".format(img_path))
            x = DataLoader().load_rgb_img(img_path, target_size=(w,h))
            x = np.reshape(x, (1, w, h, c))
            result = model.predict(x)
            print("Number of leaves: {}".format(int(result[0][0])))
        send_email("Prim info", "Prediction has finished!")
    except Exception as e:
        traceback.print_exc()
        send_email("Prim error", str(e))

if __name__ == '__main__':
    run()