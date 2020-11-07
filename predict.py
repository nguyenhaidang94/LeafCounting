from tensorflow.keras.models import load_model
from configs.global_vars import MODEL_DIR

model_name = MODEL_DIR + 'model_2020-10-31_22-59-26'

def run():
    print("Load model")
    model = load_model(model_name)
    print("Evaluate")
    result = model.evaluate()
    print(result)

if __name__ == '__main__':
    run()