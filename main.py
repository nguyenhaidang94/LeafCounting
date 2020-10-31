from configs.global_vars import BASE_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO, OPTIMIZER
from model.resnet import Resnet

def run():
    model = Resnet()
    print("Load data")
    model.load_data(BASE_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO)
    print("Build model")
    model.build()
    print("Summary")
    model.summary()
    print("Compile")
    model.compile(optimizer=OPTIMIZER)

if __name__ == '__main__':
    run()