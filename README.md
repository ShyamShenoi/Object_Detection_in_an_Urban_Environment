# Object Detection in an Urban Environment

![Inference Sample](placeholder_url_to_animation_gif)

This repository contains the complete pipeline for training, evaluating, and deploying a 2D Object Detection model on the Waymo Open Dataset. It is part of the Udacity Self-Driving Car Engineer curriculum. The goal is to successfully detect vehicles, pedestrians, and cyclists in urban driving scenarios.

## Project Structure

```
udacity-object-detection/
├── data/                    # TFRecord files (train, val, test)
├── experiments/             # Configuration files and exported models
│   └── pretrained_model/    # Checkprints and pipeline.config
├── scripts/                 # Core python modules
│   ├── download_process.py  # Script to download and pre-process Waymo dataset
│   ├── create_splits.py     # Script to split tfrecords into train/val
│   ├── edit_config.py       # Script to inject hyperparameters into pipeline config
│   ├── inference_video.py   # Script to infer on video and render animations
│   └── utils.py
├── notebooks/               # Jupyter Notebooks for EDA and demonstration
└── label_map.pbtxt          # Object class mapping
```

## Setup Instructions

1. **Environment Initialization:**
   Install all dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have installed the TensorFlow Object Detection API.

2. **Data Downloading and Splitting:**
   Download the TFRecords using the provided download script, and split them:
   ```bash
   python scripts/download_process.py --data_dir data/waymo
   python scripts/create_splits.py --source data/waymo/processed --destination data/
   ```

3. **Configure Training Pipeline:**
   Edit the pipeline configuration programmatically using the script:
   ```bash
   python scripts/edit_config.py \
       --train_dir data/train \
       --eval_dir data/val \
       --batch_size 8 \
       --checkpoint experiments/pretrained_model/checkpoint/ckpt-0 \
       --label_map label_map.pbtxt \
       --config_path experiments/pretrained_model/pipeline.config \
       --output_path experiments/pretrained_model/pipeline.config
   ```

4. **Training & Evaluation:**
   Execute the standard TF2 Object Detection model main script:
   ```bash
   python scripts/model_main_tf2.py --pipeline_config_path=experiments/pretrained_model/pipeline.config --model_dir=experiments/pretrained_model --alsologtostderr
   ```

## Exploratory Data Analysis & Inference
Check out `notebooks/Exploratory_Data_Analysis.ipynb` for data distribution understanding, and `notebooks/Inference_and_Animation.ipynb` for instructions on visualizing bounding box predictions over test scenes.
