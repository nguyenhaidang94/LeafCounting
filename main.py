from configs.global_vars import BASE_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO, OPTIMIZER
from model.resnet import Resnet

def run():
    model = Resnet()
    model.load_data(BASE_DIR, SUB_DIRS, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO)
    model.build()
    model.summary()
    model.compile(optimizer=OPTIMIZER)

if __name__ == '__main__':
    run()