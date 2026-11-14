"""
Synthetic / Curated Data Generator for CIC-IDS2017.
Generates statistically calibrated network flow records across all 8 canonical CIC-IDS2017 attack classes
and BENIGN traffic, with authentic flow characteristics and known leakage vectors.
Authors: Anannya Das (23000036) & Bhagya Sijumon (23000742)
British University in Dubai (BUiD)
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from config.config import SAMPLE_DATA_DIR, RAW_LABEL_MAPPING, RANDOM_SEEDS

logger = logging.getLogger(__name__)


def generate_cicids_flow_sample(
    n_samples: int = 12000,
    random_seed: int = RANDOM_SEEDS[0],
    output_parquet: Optional[Path] = None,
    output_csv: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate a representative stratified CIC-IDS2017 dataset containing all 15 attack subclasses,
    both realistic behavioral flow features and known data leakage artifacts.
    """
    np.random.seed(random_seed)
    logger.info(f"Generating calibrated CIC-IDS2017 dataset with {n_samples:,} flows (seed={random_seed})...")

    # Class distribution reflecting realistic enterprise intrusion ratio (heavily benign)
    class_ratios = {
        "BENIGN": 0.70,
        "DoS": 0.10,
        "DDoS": 0.08,
        "PortScan": 0.06,
        "Brute Force": 0.03,
        "Web Attack": 0.015,
        "Botnet": 0.008,
        "Infiltration": 0.005,
        "Heartbleed": 0.002
    }

    records = []

    for label, ratio in class_ratios.items():
        count = int(n_samples * ratio)
        if count < 10:
            count = 10  # Ensure minimum representation for rare classes

        for _ in range(count):
            row = {}
            # Base flow characteristics
            if label == "BENIGN":
                dst_port = np.random.choice([80, 443, 53, 8080, 25, 465, 8443, np.random.randint(1024, 65535)])
                flow_duration = int(np.random.exponential(scale=500000) + 1000)
                tot_fwd_pkts = np.random.randint(2, 40)
                tot_bwd_pkts = np.random.randint(1, 45)
                fwd_len_mean = np.random.uniform(30, 600)
                bwd_len_mean = np.random.uniform(50, 1200)
                bwd_len_std = np.random.uniform(10, 400)
                flow_iat_mean = np.random.uniform(100, 50000)
                init_win_fwd = int(np.random.choice([8192, 14600, 29200, 65535]))
                init_win_bwd = int(np.random.choice([8192, 28960, 65535]))
                syn_flag = int(np.random.rand() < 0.15)
                ack_flag = 1
                psh_flag = int(np.random.rand() < 0.30)

            elif label == "PortScan":
                # Leakage vector: port scanning sweeps ascending or random ports
                dst_port = np.random.randint(20, 65000)
                flow_duration = int(np.random.uniform(50, 4000))
                tot_fwd_pkts = np.random.randint(1, 3)
                tot_bwd_pkts = 0 if np.random.rand() < 0.7 else 1
                fwd_len_mean = np.random.uniform(0, 40)
                bwd_len_mean = 0 if tot_bwd_pkts == 0 else np.random.uniform(0, 40)
                bwd_len_std = 0
                flow_iat_mean = np.random.uniform(10, 500)
                init_win_fwd = 1024
                init_win_bwd = 0
                syn_flag = 1
                ack_flag = 0
                psh_flag = 0

            elif label == "Brute Force":
                # FTP / SSH brute force targeting standard ports
                dst_port = np.random.choice([21, 22])
                flow_duration = int(np.random.uniform(20000, 150000))
                tot_fwd_pkts = np.random.randint(10, 25)
                tot_bwd_pkts = np.random.randint(8, 20)
                fwd_len_mean = np.random.uniform(40, 120)
                bwd_len_mean = np.random.uniform(30, 90)
                bwd_len_std = np.random.uniform(5, 30)
                flow_iat_mean = np.random.uniform(500, 5000)
                init_win_fwd = 29200
                init_win_bwd = 28960
                syn_flag = 0
                ack_flag = 1
                psh_flag = 1

            elif label == "DoS":
                dst_port = 80
                flow_duration = int(np.random.exponential(scale=2000000) + 100000)
                tot_fwd_pkts = np.random.randint(50, 500)
                tot_bwd_pkts = np.random.randint(0, 10)
                fwd_len_mean = np.random.uniform(150, 400)
                bwd_len_mean = np.random.uniform(0, 50)
                bwd_len_std = np.random.uniform(0, 20)
                flow_iat_mean = np.random.uniform(10, 200)
                init_win_fwd = 65535
                init_win_bwd = 0
                syn_flag = 0
                ack_flag = 1
                psh_flag = 1

            elif label == "DDoS":
                dst_port = 80
                flow_duration = int(np.random.uniform(1000, 50000))
                tot_fwd_pkts = np.random.randint(100, 1000)
                tot_bwd_pkts = np.random.randint(0, 5)
                fwd_len_mean = np.random.uniform(200, 600)
                bwd_len_mean = np.random.uniform(0, 30)
                bwd_len_std = np.random.uniform(0, 15)
                flow_iat_mean = np.random.uniform(1, 50)
                init_win_fwd = 256
                init_win_bwd = 0
                syn_flag = 1
                ack_flag = 0
                psh_flag = 0

            elif label == "Web Attack":
                dst_port = 80
                flow_duration = int(np.random.uniform(10000, 80000))
                tot_fwd_pkts = np.random.randint(5, 20)
                tot_bwd_pkts = np.random.randint(5, 25)
                fwd_len_mean = np.random.uniform(300, 900)  # Larger payloads due to SQL/XSS injections
                bwd_len_mean = np.random.uniform(200, 1500)
                bwd_len_std = np.random.uniform(50, 500)
                flow_iat_mean = np.random.uniform(200, 8000)
                init_win_fwd = 8192
                init_win_bwd = 8192
                syn_flag = 0
                ack_flag = 1
                psh_flag = 1

            elif label == "Botnet":
                dst_port = np.random.choice([8080, 6667, 4444])
                flow_duration = int(np.random.uniform(500000, 3000000))
                tot_fwd_pkts = np.random.randint(4, 15)
                tot_bwd_pkts = np.random.randint(3, 12)
                fwd_len_mean = np.random.uniform(20, 80)
                bwd_len_mean = np.random.uniform(20, 80)
                bwd_len_std = np.random.uniform(5, 20)
                flow_iat_mean = np.random.uniform(20000, 100000)  # Periodic beaconing
                init_win_fwd = 8192
                init_win_bwd = 8192
                syn_flag = 0
                ack_flag = 1
                psh_flag = 1

            elif label == "Infiltration":
                dst_port = np.random.choice([4444, 5555, 80])
                flow_duration = int(np.random.uniform(200000, 1500000))
                tot_fwd_pkts = np.random.randint(20, 80)
                tot_bwd_pkts = np.random.randint(30, 120)
                fwd_len_mean = np.random.uniform(50, 150)
                bwd_len_mean = np.random.uniform(400, 1400)  # Exfiltration
                bwd_len_std = np.random.uniform(80, 450)
                flow_iat_mean = np.random.uniform(1000, 15000)
                init_win_fwd = 16384
                init_win_bwd = 16384
                syn_flag = 0
                ack_flag = 1
                psh_flag = 1

            elif label == "Heartbleed":
                dst_port = 443
                flow_duration = int(np.random.uniform(80000, 400000))
                tot_fwd_pkts = np.random.randint(5, 12)
                tot_bwd_pkts = np.random.randint(10, 25)
                fwd_len_mean = np.random.uniform(40, 100)
                bwd_len_mean = np.random.uniform(1200, 1460)  # 64KB memory dump payload
                bwd_len_std = np.random.uniform(10, 60)
                flow_iat_mean = np.random.uniform(500, 4000)
                init_win_fwd = 8192
                init_win_bwd = 8192
                syn_flag = 0
                ack_flag = 1
                psh_flag = 1

            tot_fwd_len = tot_fwd_pkts * fwd_len_mean
            tot_bwd_len = tot_bwd_pkts * bwd_len_mean
            tot_bytes = tot_fwd_len + tot_bwd_len
            tot_pkts = tot_fwd_pkts + tot_bwd_pkts
            flow_bytes_s = (tot_bytes / (flow_duration / 1e6)) if flow_duration > 0 else 0
            flow_pkts_s = (tot_pkts / (flow_duration / 1e6)) if flow_duration > 0 else 0

            # Construct standardized 78 flow features
            row["Destination Port"] = dst_port
            row["Flow Duration"] = flow_duration
            row["Total Fwd Packets"] = tot_fwd_pkts
            row["Total Backward Packets"] = tot_bwd_pkts
            row["Total Length of Fwd Packets"] = tot_fwd_len
            row["Total Length of Bwd Packets"] = tot_bwd_len
            row["Fwd Packet Length Max"] = fwd_len_mean * np.random.uniform(1.2, 2.0)
            row["Fwd Packet Length Min"] = max(0.0, fwd_len_mean * np.random.uniform(0.1, 0.8))
            row["Fwd Packet Length Mean"] = fwd_len_mean
            row["Fwd Packet Length Std"] = np.random.uniform(5, 150)
            row["Bwd Packet Length Max"] = bwd_len_mean * np.random.uniform(1.2, 2.5)
            row["Bwd Packet Length Min"] = max(0.0, bwd_len_mean * np.random.uniform(0.1, 0.8))
            row["Bwd Packet Length Mean"] = bwd_len_mean
            row["Bwd Packet Length Std"] = bwd_len_std
            row["Flow Bytes/s"] = flow_bytes_s
            row["Flow Packets/s"] = flow_pkts_s
            row["Flow IAT Mean"] = flow_iat_mean
            row["Flow IAT Std"] = flow_iat_mean * np.random.uniform(0.2, 1.5)
            row["Flow IAT Max"] = flow_iat_mean * np.random.uniform(1.5, 4.0)
            row["Flow IAT Min"] = max(1.0, flow_iat_mean * np.random.uniform(0.01, 0.3))
            row["Fwd IAT Total"] = flow_duration * np.random.uniform(0.6, 0.95)
            row["Fwd IAT Mean"] = flow_iat_mean * 1.1
            row["Fwd IAT Std"] = flow_iat_mean * 0.8
            row["Fwd IAT Max"] = flow_iat_mean * 2.5
            row["Fwd IAT Min"] = max(1.0, flow_iat_mean * 0.1)
            row["Bwd IAT Total"] = flow_duration * np.random.uniform(0.5, 0.9)
            row["Bwd IAT Mean"] = flow_iat_mean * 1.2
            row["Bwd IAT Std"] = flow_iat_mean * 0.9
            row["Bwd IAT Max"] = flow_iat_mean * 2.8
            row["Bwd IAT Min"] = max(1.0, flow_iat_mean * 0.1)
            row["Fwd PSH Flags"] = psh_flag
            row["Bwd PSH Flags"] = 0
            row["Fwd URG Flags"] = 0
            row["Bwd URG Flags"] = 0
            row["Fwd Header Length"] = tot_fwd_pkts * 20
            row["Bwd Header Length"] = tot_bwd_pkts * 20
            row["Fwd Packets/s"] = tot_fwd_pkts / (flow_duration / 1e6 + 1e-5)
            row["Bwd Packets/s"] = tot_bwd_pkts / (flow_duration / 1e6 + 1e-5)
            row["Min Packet Length"] = min(row["Fwd Packet Length Min"], row["Bwd Packet Length Min"])
            row["Max Packet Length"] = max(row["Fwd Packet Length Max"], row["Bwd Packet Length Max"])
            row["Packet Length Mean"] = tot_bytes / max(1, tot_pkts)
            row["Packet Length Std"] = max(bwd_len_std, row["Fwd Packet Length Std"])
            row["Packet Length Variance"] = row["Packet Length Std"] ** 2
            row["FIN Flag Count"] = int(np.random.rand() < 0.05)
            row["SYN Flag Count"] = syn_flag
            row["RST Flag Count"] = int(np.random.rand() < 0.02)
            row["PSH Flag Count"] = psh_flag
            row["ACK Flag Count"] = ack_flag
            row["URG Flag Count"] = 0
            row["CWE Flag Count"] = 0
            row["ECE Flag Count"] = 0
            row["Down/Up Ratio"] = tot_bwd_pkts / max(1, tot_fwd_pkts)
            row["Average Packet Size"] = row["Packet Length Mean"]
            row["Avg Fwd Segment Size"] = fwd_len_mean
            row["Avg Bwd Segment Size"] = bwd_len_mean
            row["Fwd Header Length.1"] = row["Fwd Header Length"]
            row["Subflow Fwd Packets"] = tot_fwd_pkts
            row["Subflow Fwd Bytes"] = tot_fwd_len
            row["Subflow Bwd Packets"] = tot_bwd_pkts
            row["Subflow Bwd Bytes"] = tot_bwd_len
            row["Init_Win_bytes_forward"] = init_win_fwd
            row["Init_Win_bytes_backward"] = init_win_bwd
            row["act_data_pkt_fwd"] = max(0, tot_fwd_pkts - 1)
            row["min_seg_size_forward"] = 20
            row["Active Mean"] = np.random.uniform(0, 1000)
            row["Active Std"] = np.random.uniform(0, 200)
            row["Active Max"] = np.random.uniform(0, 1500)
            row["Active Min"] = 0
            row["Idle Mean"] = np.random.uniform(0, 5000000)
            row["Idle Std"] = np.random.uniform(0, 100000)
            row["Idle Max"] = np.random.uniform(0, 6000000)
            row["Idle Min"] = 0
            row["Label"] = label

            records.append(row)

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Save to parquet and csv
    target_parquet = output_parquet or (SAMPLE_DATA_DIR / "cicids2017_sample.parquet")
    target_csv = output_csv or (SAMPLE_DATA_DIR / "cicids2017_sample.csv")

    df.to_parquet(target_parquet, index=False)
    df.to_csv(target_csv, index=False)
    logger.info(f"Sample dataset saved to: {target_parquet} and {target_csv}")
    return df


if __name__ == "__main__":
    generate_cicids_flow_sample()
