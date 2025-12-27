import os
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("/app/backend/ml/datasets/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs
DATASET_URLS = {
    "CICIDS2017": {
        "urls": [
            "https://www.unb.ca/cic/datasets/ids-2017.html"
        ],
        "manual": True,
        "description": "CICIDS2017 requires manual download from UNB website"
    },
    "UNSW-NB15": {
        "urls": [
            "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download",
        ],
        "manual": False
    },
    "NSL-KDD": {
        "urls": [
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        ],
        "manual": False
    },
    "KDDCup99": {
        "urls": [
            "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        ],
        "manual": False
    },
    "CTU-13": {
        "urls": [
            "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/"
        ],
        "manual": True,
        "description": "CTU-13 requires manual download"
    }
}


def download_file(url: str, destination: Path):
    """Download a file from URL to destination"""
    try:
        logger.info(f"Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded to {destination}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def extract_gz(gz_path: Path, output_path: Path):
    """Extract .gz file"""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Extracted {gz_path} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error extracting {gz_path}: {e}")
        return False


def download_nsl_kdd():
    """Download NSL-KDD dataset"""
    dataset_dir = DATA_DIR / "NSL-KDD"
    dataset_dir.mkdir(exist_ok=True)
    
    files = [
        ("KDDTrain+.txt", DATASET_URLS["NSL-KDD"]["urls"][0]),
        ("KDDTest+.txt", DATASET_URLS["NSL-KDD"]["urls"][1]),
    ]
    
    for filename, url in files:
        dest = dataset_dir / filename
        if not dest.exists():
            download_file(url, dest)
        else:
            logger.info(f"{filename} already exists")
    
    return dataset_dir


def download_kddcup99():
    """Download KDDCup99 dataset"""
    dataset_dir = DATA_DIR / "KDDCup99"
    dataset_dir.mkdir(exist_ok=True)
    
    gz_file = dataset_dir / "kddcup.data_10_percent.gz"
    csv_file = dataset_dir / "kddcup.data_10_percent.csv"
    
    if not csv_file.exists():
        if not gz_file.exists():
            download_file(DATASET_URLS["KDDCup99"]["urls"][0], gz_file)
        extract_gz(gz_file, csv_file)
    else:
        logger.info("KDDCup99 already exists")
    
    return dataset_dir


def download_all_datasets():
    """Download all available datasets"""
    logger.info("Starting dataset downloads...")
    
    # Download NSL-KDD
    download_nsl_kdd()
    
    # Download KDDCup99
    download_kddcup99()
    
    logger.info("\n=== MANUAL DOWNLOAD REQUIRED ===")
    logger.info("Please manually download:")
    logger.info("1. CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
    logger.info("2. UNSW-NB15 from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    logger.info("3. CTU-13 from: https://www.stratosphereips.org/datasets-ctu13")
    logger.info(f"Place files in: {DATA_DIR}")
    
    return DATA_DIR


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all_datasets()