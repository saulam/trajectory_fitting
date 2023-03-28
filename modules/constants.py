"""
Constants.
"""

__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve'
__email__ = "saul.alonso.monsalve@cern.ch"

# Training dataset path
TRAINING_DATASET = "events"

# for all the methods
DETECTOR_RANGES = ((-1000., +1000.),
                   (-1000., +1000.),
                   (-1000., +1000.))
CHARGE_RANGE = (0., 500)

# for SIR-PF only
HIST_PATH = "/scratch2/salonso/particle_filtering/general_sample/training/histogram.pkl"
IDX_SPOS = 0
IDX_EPOS = IDX_SPOS+3
IDX_SCHA = IDX_EPOS
IDX_ECHA = IDX_SCHA+1
IDX_SVAR = IDX_EPOS
IDX_EVAR = IDX_SVAR+3
UNIT_MM = 1.0
UNIT_CM = 10.*UNIT_MM
CUBE_SIZE = (10./2)*UNIT_MM

# training
NUM_EPOCHS = 500
BATCH_SIZE = 128
PAD_IDX = -1
INPUT_SIZE = 4
OUTPUT_SIZE = 3
D_MODEL = 64
N_HEAD = 8
DIM_FEEDFORWARD = 128
NUM_ENCODER_LAYERS = 5
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9
EARLY_STOPPING = 30
