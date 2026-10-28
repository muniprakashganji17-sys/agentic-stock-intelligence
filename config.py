import os
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]

START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"

TRAIN_END = "2022-12-31"   # train: 2018-2022
VAL_END   = "2023-12-31"   # validation: 2023
                           # test: 2024-2025

OLLAMA_MODEL = "qwen2.5:3b"
DATA_ROOT     = os.environ.get("ASI_DATA_ROOT", "data")
RAW_DIR       = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")