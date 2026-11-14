# CIC-IDS2017 Dataset Acquisition & Setup Guide

This project includes a fully calibrated, stratified sample dataset in `data/sample/cicids2017_sample.parquet` and `data/sample/cicids2017_sample.csv` so you can train models, execute tests, and launch the interactive dashboard immediately out-of-the-box.

If you wish to benchmark on the full ~2.83 million flows, follow the steps below.

---

## 1. Official Dataset Sources

1. **Kaggle Parquet Version (Recommended)**:
   - URL: [https://www.kaggle.com/datasets/dhoogla/cicids2017](https://www.kaggle.com/datasets/dhoogla/cicids2017)
   - Format: Single pre-merged Parquet file or daily Parquet splits.
   - Size: ~300 MB compressed / 2.5 GB uncompressed.

2. **University of New Brunswick (UNB) Original**:
   - URL: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
   - Format: 8 daily CSV files (`Monday-WorkingHours.pcap_ISCX.csv` to `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`).

---

## 2. Placing Data in Project

Download the parquet or CSV file and place it in the `data/` directory:
```
data/
├── cicids2017_full.parquet    <--- Place full dataset here
└── sample/
    ├── cicids2017_sample.parquet
    └── cicids2017_sample.csv
```

---

## 3. Running Benchmark on Full Dataset

To train and evaluate on the full dataset, supply the file path to the pipeline:
```bash
python -m src.experiments.run_all --data-path data/cicids2017_full.parquet --sample-frac 0.25
```
*(Tip: Setting `--sample-frac 0.25` provides a 700,000 flow sample which balances maximum statistical power with rapid training turnaround).*
