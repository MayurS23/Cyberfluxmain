import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("/app/backend/ml/datasets/raw")
PROCESSED_DIR = Path("/app/backend/ml/datasets/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# NSL-KDD column names
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]


class DatasetPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def preprocess_nsl_kdd(self):
        """Preprocess NSL-KDD dataset"""
        logger.info("Preprocessing NSL-KDD...")
        
        train_path = DATA_DIR / "NSL-KDD" / "KDDTrain+.txt"
        test_path = DATA_DIR / "NSL-KDD" / "KDDTest+.txt"
        
        if not train_path.exists():
            logger.warning(f"NSL-KDD train file not found at {train_path}")
            return None
        
        # Load data
        df_train = pd.read_csv(train_path, names=NSL_KDD_COLUMNS, header=None)
        df_test = pd.read_csv(test_path, names=NSL_KDD_COLUMNS, header=None)
        
        # Combine for processing
        df = pd.concat([df_train, df_test], ignore_index=True)
        
        # Remove difficulty column
        df = df.drop('difficulty', axis=1)
        
        # Binary classification: normal vs attack
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        self.feature_columns = list(X.columns)
        
        # Save processed data
        output_path = PROCESSED_DIR / "nsl_kdd_processed.pkl"
        joblib.dump({
            'X': X_scaled,
            'y': y,
            'feature_names': self.feature_columns
        }, output_path)
        
        logger.info(f"NSL-KDD processed: {len(X_scaled)} samples, {len(self.feature_columns)} features")
        return X_scaled, y
    
    def preprocess_kddcup99(self):
        """Preprocess KDDCup99 dataset"""
        logger.info("Preprocessing KDDCup99...")
        
        data_path = DATA_DIR / "KDDCup99" / "kddcup.data_10_percent.csv"
        
        if not data_path.exists():
            logger.warning(f"KDDCup99 file not found at {data_path}")
            return None
        
        # KDDCup99 uses same columns as NSL-KDD (minus difficulty)
        columns = NSL_KDD_COLUMNS[:-1]  # Remove 'difficulty'
        
        df = pd.read_csv(data_path, names=columns, header=None)
        
        # Clean label column (remove trailing dot)
        df['label'] = df['label'].str.rstrip('.')
        
        # Binary classification
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                df[col] = df[col].map(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ else -1
                )
        
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )
        
        # Save processed data
        output_path = PROCESSED_DIR / "kddcup99_processed.pkl"
        joblib.dump({
            'X': X_scaled,
            'y': y,
            'feature_names': list(X.columns)
        }, output_path)
        
        logger.info(f"KDDCup99 processed: {len(X_scaled)} samples")
        return X_scaled, y
    
    def create_unified_dataset(self):
        """Create unified dataset from all sources"""
        logger.info("Creating unified dataset...")
        
        datasets = []
        
        # Preprocess individual datasets
        nsl_data = self.preprocess_nsl_kdd()
        if nsl_data:
            datasets.append(nsl_data)
        
        kdd_data = self.preprocess_kddcup99()
        if kdd_data:
            datasets.append(kdd_data)
        
        if not datasets:
            logger.error("No datasets available for processing")
            return None, None
        
        # Combine all datasets
        X_combined = pd.concat([data[0] for data in datasets], ignore_index=True)
        y_combined = pd.concat([data[1] for data in datasets], ignore_index=True)
        
        # Save unified dataset
        output_path = PROCESSED_DIR / "unified_dataset.pkl"
        joblib.dump({
            'X': X_combined,
            'y': y_combined,
            'feature_names': self.feature_columns,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, output_path)
        
        logger.info(f"Unified dataset created: {len(X_combined)} samples")
        logger.info(f"Normal: {(y_combined == 0).sum()}, Attack: {(y_combined == 1).sum()}")
        
        return X_combined, y_combined
    
    def save_preprocessor(self):
        """Save scaler and encoders"""
        preprocessor_path = PROCESSED_DIR / "preprocessor.pkl"
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = DatasetPreprocessor()
    X, y = preprocessor.create_unified_dataset()
    preprocessor.save_preprocessor()