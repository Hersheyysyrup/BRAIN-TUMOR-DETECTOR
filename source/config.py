###holds the image size (128), batch size, seed,
###validation fraction, learning rates, epochs, and all the folder paths

"""Paths, seed and img_formats are defined in data.py"""

from data import DATASET, IMG_FORMATS, ROOT, SEED, TEST_DIR, TRAIN_DIR

IMG_SIZE = 244    ###to be changes 128 later 
BATCH_SIZE = 32
VAL_SIZE = 0.15

EPOCHS = 8
FINE_TUNE_EPOCHS = 10
HEAD_LR = 1e-3   #0.001
FINE_TUNE_LR = 1e-5    #0.00001

MODELS_DIR = ROOT/"models"
CHECKPOINT_DIR = MODELS_DIR/"checkpoints"
RESULTS_DIR = ROOT/"results"
PLOTS_DIR = RESULTS_DIR/"plots"
METRICS_DIR = RESULTS_DIR/"metrics"

def make_dirs():
    """for creating output folders if they dont exists"""
    for d in (CHECKPOINT_DIR, PLOTS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)