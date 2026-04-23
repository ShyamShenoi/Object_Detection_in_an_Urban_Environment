import argparse
import glob
import os
import random
import shutil

from utils import get_module_logger

def split(source_dir, destination_dir, split_ratio=(0.8, 0.2)):
    """
    Randomly splits all `.tfrecord` files from `source_dir` into `train` and `val`
    directories inside `destination_dir`. (Test set is kept out to simplify the pipeline,
    as typical in Udacity Project unless a test set split is desired).
    """
    logger = get_module_logger(__name__)
    
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        logger.error(f"Source directory {source_dir} does not exist.")
        return

    # Find all tfrecords
    tfrecords = glob.glob(os.path.join(source_dir, "*.tfrecord"))
    if not tfrecords:
        logger.error(f"No .tfrecord files found in {source_dir}")
        return
        
    logger.info(f"Found {len(tfrecords)} TFRecord files.")
    
    # Sort and shuffle to ensure reproducibility while mixing chronological runs
    tfrecords.sort()
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(tfrecords)

    # Calculate split index
    train_ratio = split_ratio[0]
    num_train = int(len(tfrecords) * train_ratio)
    
    train_files = tfrecords[:num_train]
    val_files = tfrecords[num_train:]
    
    # Define destination folders
    train_dir = os.path.join(destination_dir, "train")
    val_dir = os.path.join(destination_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    logger.info(f"Copying {len(train_files)} files to {train_dir}")
    for file_path in train_files:
        dest_path = os.path.join(train_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest_path)
        
    logger.info(f"Copying {len(val_files)} files to {val_dir}")
    for file_path in val_files:
        dest_path = os.path.join(val_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest_path)
        
    logger.info("Dataset splitting complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split TFRecord files into Train/Val sets.")
    parser.add_argument("--source", required=True, type=str, help="Source directory containing .tfrecord files")
    parser.add_argument("--destination", required=True, type=str, help="Target directory where train/val folders will be created")
    
    args = parser.parse_args()
    split(args.source, args.destination)
