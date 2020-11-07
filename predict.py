import os
from tensorflow.keras.models import load_model
from configs.global_vars import MODEL_DIR, DATA_DIR

def run():
    print("Load model")
    model_name = os.path.join(MODEL_DIR, 'model_2020-10-31_22-59-26')
    model = load_model(model_name)
    print("Predict")
    img_path = os.path.join(DATA_DIR, 'A1', 'plant001_rgb.png')
    result = model.predict(img_path)
    print(result)

if __name__ == '__main__':
    run()