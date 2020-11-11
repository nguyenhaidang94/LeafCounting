import traceback
from configs.global_vars import DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO\
    , OPTIMIZER, BATCH_SIZE, N_EPOCHS, ES_EPOCHS
from model.resnet import Resnet
from utils.mail_utils import send_email

def run():
    try:
        model = Resnet()
        print("Load data")
        model.load_data(DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO)
        print("Build model")
        model.build(train_base_model=True)
        print("Summary")
        model.summary()
        print("Compile model")
        model.compile(optimizer=OPTIMIZER)
        print("Train model")
        model.train(BATCH_SIZE, N_EPOCHS, ES_EPOCHS)
        print("Evaluate on test set")
        result = model.evaluate()
        print("Loss: {}".format(result))
        print("Save model")
        model.save()
        send_email("Prim info", "Training has finished!")
    except Exception as e:
        traceback.print_exc()
        send_email("Prim error", str(e))

if __name__ == '__main__':
    run()