from configs.global_vars import DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO\
    , OPTIMIZER, BATCH_SIZE, N_EPOCHS
from model.resnet import Resnet

def run():
    model = Resnet()
    print("Load data")
    model.load_data(DATA_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO)
    print("Build model")
    model.build()
    print("Summary")
    model.summary()
    print("Compile model")
    model.compile(optimizer=OPTIMIZER)
    print("Train model")
    model.train(BATCH_SIZE, N_EPOCHS)
    print("Evaluate on test set")
    result = model.evaluate()
    print("Loss: {}".format(result))
    print("Save model")
    model.save()

if __name__ == '__main__':
    run()