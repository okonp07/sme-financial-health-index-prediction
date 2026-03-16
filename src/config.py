from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
METRICS_DIR = OUTPUTS_DIR / "metrics"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

TRAIN_PATH = RAW_DATA_DIR / "Train.csv"
TEST_PATH = RAW_DATA_DIR / "Test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "SampleSubmission.csv"
VARIABLE_DEFINITIONS_PATH = RAW_DATA_DIR / "VariableDefinitions.csv"

TARGET_COLUMN = "Target"
ID_COLUMN = "ID"
RANDOM_STATE = 42
N_SPLITS = 5
