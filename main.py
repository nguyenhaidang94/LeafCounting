import traceback
import numpy as np
from configs.global_vars import DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO\
    , OPTIMIZER, BATCH_SIZE, N_EPOCHS, TRAIN_BASE_MODEL, ES_EPOCHS
from model.resnet import Resnet
from utils.mail_utils import send_email

N = 3

def run():
    try:
        models = []
        losses = []
        for i in range(N):
            model = Resnet()
            print("Load data")
            model.load_data(DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO)
            print("Build model")
            model.build(train_base_model=TRAIN_BASE_MODEL)
            print("Compile model")
            model.compile(optimizer=OPTIMIZER)
            print("Train model")
            model.train(BATCH_SIZE, N_EPOCHS, ES_EPOCHS)
            print("Evaluate on test set")
            loss = model.evaluate()
            print("Loss: {}".format(loss))
            losses.append(loss)
            models.append(model)
        print("Mean loss: {}".format(np.mean(losses)))
        print("Use the best model to evaluate")
        best_model_idx = np.argmin(losses)
        models[best_model_idx].evaluate()
        print("Save model")
        models[best_model_idx].save()
        send_email("Prim info", "Training has finished!")
    except Exception as e:
        traceback.print_exc()
        send_email("Prim error", str(e))

if __name__ == '__main__':
    run()