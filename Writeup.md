# Writeup: Object Detection in an Urban Environment

## 1. Goal
The primary objective is to take raw autonomous driving camera data provided by the Waymo Open Dataset, process it into a format understandable by TensorFlow (TFRecords), perform exploratory data analysis, and then fine-tune a pre-trained convolutional neural network (SSD ResNet50 V1 FPN) for accurate 2D bounding box prediction.

## 2. Dataset Processing and EDA
Initially, the data distributions were heavily skewed. By visualizing the data in the `notebooks/Exploratory_Data_Analysis.ipynb` notebook, it became evident that the number of `Vehicles` significantly outweighed `Pedestrians` and `Cyclists`. This imbalance dictated our need for heavy data augmentations. 

We split the processed TFRecord files randomly with an 80/20 train/validation split using the custom `scripts/create_splits.py` file, preserving temporal and geographic sequence dependencies.

## 3. Model Architecture Selection
I selected the **SSD ResNet50 V1 FPN** (Single Shot MultiBox Detector with a Feature Pyramid Network). 
- **Reasoning:** SSD architectures offer a phenomenal tradeoff between inference speed and accuracy, crucial for real-time autonomous driving applications. The addition of the Feature Pyramid Network allows the model to handle objects of varying scales extremely well (e.g., pedestrians far away vs. close).
- The `pipeline.config` was heavily customized using `scripts/edit_config.py`. 

## 4. Hyperparameter Tuning & Cross-Validation
Through automated config editing, the following hyperparameters were fine-tuned:
- **Batch Size:** Maintained at 8 to allow stable gradient descent given standard VRAM constraints.
- **Learning Rate:** Utilized an initial learning rate of `0.04` with a warmup and cosine decay scheduler.
- **Data Augmentation:** The configuration was updated to apply:
  1. Random Horizontal Flips (To ensure left-right spatial invariance)
  2. Random Crop (To simulate zoom and prevent overfitting to central objects)

## 5. Results and Evaluation
The model achieved robust convergence. The training localization and classification losses steadily decreased. When evaluated against the validation bounding boxes, the model demonstrated excellent recall for vehicles, though recall on cyclists remained lower due to dataset class imbalance.

Visualizations of the trained model over validation sequences (`Inference_and_Animation.ipynb`) demonstrate stable bounding box tracking over time.
