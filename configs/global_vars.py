from tensorflow.keras.optimizers import Adam

DATA_DIR = "/ldaphome/dhai/prim/CVPPP2017_LCC_training"
MODEL_DIR = "/ldaphome/dhai/prim/models"
SUB_DIRS = ['A1', 'A2', 'A3', 'A4']
IMAGE_SIZE = (320, 320, 3)
TRAIN_RATIO = 0.5
VAL_RATIO = 0.25
LEARNING_RATE = 1e-3
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
BATCH_SIZE = 32
N_EPOCHS = 50
# for data augmentation
ROTATION_RANGE = 170
ZOOM_RANGE = [1.0, 1.1]
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True