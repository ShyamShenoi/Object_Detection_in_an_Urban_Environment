# Personal Learning Guide: Object Detection in an Urban Environment 🚗

This document serves as an exhaustive, step-by-step masterclass on how this project operates. It is tailored for beginners to deeply understand the algorithms, the codebase, and the exact terminal commands required to run the project and push it to version control.

---

## 🧠 1. The Algorithm: How Does It Work?

For this model, we are relying on an architecture called **SSD ResNet50 V1 FPN**. 

1. **SSD (Single Shot MultiBox Detector):** 
   - Traditional object detectors like R-CNN use a two-step process: first, they draw hundreds of random boxes guessing where an object *might* be (Region Proposals), and then they feed every single box to a classifier. 
   - SSD is "Single Shot." It proposes bounding boxes and classifies what is inside them simultaneously in one single forward pass of the network. This makes it incredibly fast and ideal for real-time autonomous driving where latency is fatal.

2. **ResNet50 Backbone:** 
   - The "brain" of our model is a 50-layer deep Convolutional Neural Network (CNN) called ResNet. This network has been pre-trained on millions of standard images (like dogs, cats, and cars). It acts as our *feature extractor*—scanning pixels to identify edges, tires, arms, and shapes.

3. **FPN (Feature Pyramid Network):** 
   - A common weakness of fast SSD models is detecting very small objects (like pedestrians far away in a dashcam). 
   - FPN solves this by analyzing the image at multiple scales (a "pyramid" of resolutions). It takes high-level understanding from the deep layers and combines it with the sharp pixel details from the early layers, allowing the model to spot objects of vastly different sizes at the same time.

---

## 🛠️ 2. Environment Setup & Dependencies

Before touching any code, you need a pristine Python environment. Installing packages globally usually causes conflicts.

**Step 2.1: Create a Virtual Environment**
By creating a "virtual environment," you build a sandbox where TensorFlow can be installed without breaking your computer's main Python instance.
```bash
python3 -m venv obj_det_env
source obj_det_env/bin/activate
```

**Step 2.2: Install the specific libraries**
It is critical to install the exact versions listed in our `requirements.txt`.
```bash
pip install -r requirements.txt
```
*Note: This strictly installs `tensorflow==2.10.0` and the `tf-models-official` API, ensuring the Object Detection API does not clash with the Waymo dataset parsers.*

---

## 🚀 3. Step-by-Step Codebase Execution

### Step 3.1: Download and Process the Waymo Dataset
The Waymo Open Dataset provides massive, high-definition folders of frames. Our model cannot read these directly; it needs `TFRecords`. A TFRecord is a highly compressed binary file format natively understood by TensorFlow.

**Command:**
```bash
python scripts/download_process.py --data_dir data/waymo
```
*Code Logic:* This script contacts the Waymo servers, streams the dataset, and repacks the images and their bounding box labels into `.tfrecord` hashes inside the `data/waymo/processed` directory.

### Step 3.2: Split the Data (`create_splits.py`)
In Machine Learning, we cannot test the model on the exact same images it trained on, otherwise it "memorizes" the answers (Overfitting). 

**Command:**
```bash
python scripts/create_splits.py --source data/waymo/processed --destination data/
```
*Code Logic:* This script looks at all the TFRecords we downloaded, shuffles them securely, and moves 80% of them into the `data/train` folder and 20% into `data/val` (Validation).

### Step 3.3: Exploratory Data Analysis (EDA)
Before training, data scientists must visually inspect their data to confirm there are no disastrous imbalances (e.g., millions of cars but zero pedestrians).

**Command:**
```bash
jupyter notebook notebooks/Exploratory_Data_Analysis.ipynb
```
*Action:* Run the notebook cells. It iterates through the TFRecords and uses Python's `matplotlib` to draw standard boxes around the cars and pedestrians, outputting a histogram of the classes. 

### Step 3.4: Edit the Training Configuration (`edit_config.py`)
TensorFlow's Object Detection API requires a massive instruction manual called `pipeline.config`. You don't edit it by hand; we wrote a script to do it.

**Command:**
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
*Code Logic:* The script opens `pipeline.config`, sets the Batch Size to 8 (so we don't crash our GPU memory), points the model to our `data/train` files, and adds "Data Augmentations" like randomly flipping images horizontally so the model learns that a car is a car regardless of what direction it points!

### Step 3.5: Train the Model
Now the heavy neural network training begins. TensorFlow feeds batches of images into the ResNet, calculates how badly it guessed the bounding boxes, and adjusts its internal math via Backpropagation.

**Command:**
```bash
python scripts/model_main_tf2.py \
    --pipeline_config_path=experiments/pretrained_model/pipeline.config \
    --model_dir=experiments/pretrained_model \
    --alsologtostderr
```
*Note: Run this on a GPU instance, as it takes tens of thousands of "steps" to lower the loss curve.*

### Step 3.6: Export the Model and Animate
Once training is successful, your model is saved as a "Frozen Graph". We feed a raw video sequence into it to see its real-world performance!

**Command:**
```bash
jupyter notebook notebooks/Inference_and_Animation.ipynb
```
*Code Logic:* This notebook executes `inference_video.py`, which loads the Frozen Graph, passes a test video frame-by-frame into the network, draws the predicted bounding boxes, and stitches the frames together using `matplotlib.animation` into a final `.mp4` file.

---

## 🌐 4. Publishing to GitHub

Once you finish your work locally, you must "Push" your code to a remote GitHub repository. Version control ensures you never lose a script and can proudly display your portfolio.

**Warning First:** Always ensure you have a `.gitignore` file that ignores the gigantic `.tfrecord` data and `.ckpt` trained weights. GitHub blocks file uploads larger than 100MB! We already created this file for you.

**Step 1: Stage the Files**
This command packages all changes made in your folder.
```bash
git add .
```

**Step 2: Commit your files**
This labels the package of changes with a readable message.
```bash
git commit -m "Add personal Beginner Guide and document scripts"
```

**Step 3: Link to your Remote GitHub Repository**
*(You only technically run this once when setting up a fresh folder).*
```bash
git remote add origin https://github.com/ShyamShenoi/Object_Detection_in_an_Urban_Environment.git
```

**Step 4: Push to the Cloud!**
This uploads the files. (If prompted, enter your GitHub Username and Personal Access Token password).
```bash
git push -u origin main
```

🎉 **Congratulations! You now thoroughly understand the physics, the scripts, and the terminal operations of this project!**
