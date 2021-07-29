import os

# set directories
ROOT_DIR = '/notebooks/horoscope'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
MODEL_BIN = os.path.join(MODEL_DIR, 'model.bin')
DATA_FILE = os.path.join(DATA_DIR, 'horoscope_cleaned.csv')

# h5 params
H5_CHUNKSIZE = 2000
SEQUENCE_LENGTH = 300

# model params
MODEL_NAME = 'gpt2'
LEARNING_RATE = 5e-4
WARMUP_STEPS = 1e2
EPSILON = 1e-8
EPOCHS = 5
DEVICE = 'cuda'