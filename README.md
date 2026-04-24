# Object Detection in an Urban Environment 🚗

This repository contains an end-to-end tutorial and codebase for training a Machine Learning model to identify and locate vehicles, pedestrians, and cyclists in urban driving scenarios using the **TensorFlow Object Detection API**. 

Whether you are a beginner or a veteran, this document contains **every minute detail** required to understand the algorithms, configure the environment, execute the training, and push your final code to GitHub.

---

## 🧠 1. The Algorithm: How Does It Work?

For this project, we are relying on a popular architecture called **SSD ResNet50 V1 FPN**. Let's break down what that means:

1. **SSD (Single Shot MultiBox Detector):** Traditional Object Detectors (like Faster R-CNN) use a two-step process: first, they guess where an object might be (Region Proposals), and then they classify it. SSD is "Single Shot." It proposes bounding boxes and classifies them in one single pass. This makes it incredibly fast and ideal for real-time autonomous driving.
2. **ResNet50 Backbone:** The "brain" of our model is a 50-layer deep Convolutional Neural Network called ResNet. This network has been pre-trained on millions of standard images. It acts as our feature extractor—identifying edges, tires, arms, and shapes.
3. **FPN (Feature Pyramid Network):** A common weakness of SSDs is detecting very small objects (like pedestrians far away in a dashcam). FPN solves this by looking at the image at multiple scales (a "pyramid" of resolutions), combining high-level data with low-level details so the model can spot objects of all sizes.

---

## 🛠️ 2. Environment Setup & Dependencies

Before touching any code, you need a pristine Python environment. Open your command prompt/terminal and run these exact steps.

**Step A: Create a Virtual Environment (Optional but Recommended)**
```bash
python3 -m venv obj_det_env
source obj_det_env/bin/activate
```

**Step B: Install the specific libraries**
It is critical to install the exact versions listed in our `requirements.txt` to avoid compatibility errors between the Waymo dataset and TensorFlow.
```bash
pip install -r requirements.txt
```
*Note: This will install `tensorflow==2.10.0`, the `tf-models-official` API, and standard data science tools like `matplotlib` and `jupyter`.*

---

## 📁 3. Project Structure

Here is what your initialized repository looks like:
* **`data/`**: Where your raw imagery (`train`, `val`, `test`) will live.
* **`experiments/pretrained_model/`**: Where your trained model weights and configurations live.
* **`scripts/`**: The core logic scripts required for moving data and talking to TensorFlow.
* **`notebooks/`**: Interactive Jupyter notebooks to visualize your data before and after training.

---

## 🚀 4. Step-by-Step Execution Guide

### Step 4.1: Download and Process the Waymo Dataset
The Waymo Open Dataset provides thousands of high-definition frames of varying weather and lighting. We need to convert it into `TFRecords` (a highly optimized binary file format TensorFlow loves to read).

**Command Prompt:**
```bash
python scripts/download_process.py --data_dir data/waymo
```

### Step 4.2: Split the Data
We cannot train and test the model on the exact same images, otherwise it will "memorize" the answers rather than learning. We securely split our data into Training (80%) and Validation (20%) piles.

**Command Prompt:**
```bash
python scripts/create_splits.py --source data/waymo/processed --destination data/
```

### Step 4.3: Exploratory Data Analysis (EDA)
Before training, it is crucial to look at the data to understand class imbalances (e.g., if there are 1,000 cars but only 50 pedestrians).
**Command Prompt:**
```bash
jupyter notebook notebooks/Exploratory_Data_Analysis.ipynb
```
*Run the cells sequentially to see the bounding boxes drawn over actual dashcam imagery!*

### Step 4.4: Edit the Training Configuration
TensorFlow relies heavily on a `pipeline.config` file. We use a python script to automatically inject our file paths, set our Batch Size to 8, and add Data Augmentations (like randomly flipping an image horizontally so the model learns that a car is a car regardless of the direction it faces).

**Command Prompt:**
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

### Step 4.5: Train the Model
Now the heavy lifting begins. We execute the standard TensorFlow API loop. *(Warning: Running this without a GPU will be excruciatingly slow).*

**Command Prompt:**
```bash
python scripts/model_main_tf2.py \
    --pipeline_config_path=experiments/pretrained_model/pipeline.config \
    --model_dir=experiments/pretrained_model \
    --alsologtostderr
```

### Step 4.6: Export the Model and Animate
Once the loss drops sufficiently and training is complete, you freeze your graph and run it through a fresh video sequence to witness the magic!

**Command Prompt:**
```bash
# Create the animation MP4
jupyter notebook notebooks/Inference_and_Animation.ipynb
```

---

## 🌐 5. Publishing to GitHub

Once you reach the end of the project and your model is working, it's time to share your code! We will push your custom code up to a remote repository securely so others can view it.

*(Note: Ensure you have placed large `.tfrecord` files and massive `.pb` model checkpoints inside folders covered by your `.gitignore` file, otherwise GitHub will reject the push because the files are too large).*

**Step 1: Initialize Git and Stage the Files**
```bash
git init
git add .
```

**Step 2: Commit your files**
```bash
git commit -m "Initial commit: Complete Object Detection Pipeline"
```

**Step 3: Link to your Remote GitHub Repository**
Replace the URL with your exact repository address.
```bash
git remote add origin https://github.com/ShyamShenoi/Object_Detection_in_an_Urban_Environment.git
```

**Step 4: Rename branch (Optional best practice) and Push!**
*(Note: If prompted, enter your GitHub Username and Personal Access Token).*
```bash
git branch -M main
git push -u origin main
```

🎉 **Congratulations! Your code is now live and professionally documented.**
