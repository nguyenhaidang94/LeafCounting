import traceback
import numpy as np
from configs.global_vars import DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO\
    , OPTIMIZER, BATCH_SIZE, N_EPOCHS, ES_EPOCHS
from configs.global_vars import N_RUN
from model.resnet import Resnet
from utils.mail_utils import send_email

def run():
    try:
        models = []
        losses = []
        for i in range(N_RUN):
            model = Resnet()
            print("Load data")
            model.load_data(DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO)
            print("Build model")
            model.build()
            print("Compile model")
            model.compile(optimizer=OPTIMIZER)
            print("Train model")
            model.train(BATCH_SIZE, N_EPOCHS, ES_EPOCHS)
            print("Calculate loss on test set")
            loss = model.loss_in_test_set()
            print("Loss: {}".format(loss))
            losses.append(loss)
            models.append(model)
        print("Mean loss: {}".format(np.mean(losses)))
        print("Use the best model to evaluate")
        best_model_idx = np.argmin(losses)
        models[best_model_idx].evaluate()
        models[best_model_idx].save()
        send_email("Prim info", "Training has finished!")
    except Exception as e:
        traceback.print_exc()
        send_email("Prim error", str(e))

if __name__ == '__main__':
    run()