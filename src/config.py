import pathlib
import torch

# ── Đường dẫn ──────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.parent

_MUSHROOM_ROOT = BASE_DIR / "dataset" / "raw" / "Mushroom"
DATASET_DIR = _MUSHROOM_ROOT / "train"
TRAIN_DIR   = DATASET_DIR          
VAL_DIR     = _MUSHROOM_ROOT / "valid"
TEST_DIR    = _MUSHROOM_ROOT / "test"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MUSHROOM_NET_PATH  = MODEL_DIR / "mushroomnet.pth"
RESNET50_PATH      = MODEL_DIR / "resnet50.pth"
GENETIC_MODEL_PATH = MODEL_DIR / "mushroomnet_genetic.pth"

# ── 12 loài trong Open Dataset (Kaggle)─────────────────────────
CLASS_NAMES_OPEN = [
    "Agaricus", "Amanita", "Boletus", "Cortinarius",
    "Entoloma", "Exidia", "Hygrocybe", "Inocybe",
    "Lactarius", "Pluteus", "Russula", "Suillus",
]
 
# ── 18 loài trong Local Dataset ──────────────────────────────────
CLASS_NAMES_LOCAL = [
    "Amanita", "Armillaria mellea", "Boletus", "Cantharellus cibarius",
    "Collybia", "Ganoderma", "Laccaria", "Lactarius",
    "Lentinus edodes", "Morchella", "Ophiocordyceps sinensis",
    "Pleurotus citrinopileatus", "Ramaria", "Russula",
    "Sarcodon imbricatum", "Thelephora ganbajun",
    "Trichloma matsutake", "Tuber",
]

# ── Tham số ảnh ─────────────────────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)

# ── Tham số huấn luyện  ─────────────────────────
BATCH_SIZE    = 12
EPOCHS_STAGE2 = 30
EPOCHS_STAGE3 = 30
EPOCHS_GENETIC = 30
LEARNING_RATE  = 1e-4
SEED = 42

# ── Split tỉ lệ: 80% train / 10% val / 10% test ─────
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# ── Thiết bị ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
