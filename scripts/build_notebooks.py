import json
import os

def create_eda_notebook(out_path):
    eda_nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Exploratory Data Analysis: Waymo Dataset\n",
                    "This notebook demonstrates custom logic for interacting with the tfrecords, visualizing bounding boxes, and understanding object class distributions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import tensorflow as tf\n",
                    "import matplotlib.pyplot as plt\n",
                    "import matplotlib.patches as patches\n",
                    "import numpy as np\n",
                    "import glob\n",
                    "import os\n",
                    "\n",
                    "# We'll assume scripts/utils.py is available or we add it to the path\n",
                    "import sys\n",
                    "sys.path.append('../scripts')\n",
                    "from utils import get_dataset"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Helper function for visualizing instances"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def display_instances(batch, max_images=5):\n",
                    "    \"\"\"\n",
                    "    Custom implementation to extract images and bounding boxes from a parsed record batch,\n",
                    "    and plot them using matplotlib.\n",
                    "    \"\"\"\n",
                    "    images = batch['image'].numpy()\n",
                    "    boxes = batch['groundtruth_boxes'].numpy()\n",
                    "    classes = batch['groundtruth_classes'].numpy()\n",
                    "    \n",
                    "    num_images = min(len(images), max_images)\n",
                    "    fig, axes = plt.subplots(num_images, 1, figsize=(15, 8 * num_images))\n",
                    "    if num_images == 1:\n",
                    "        axes = [axes]\n",
                    "        \n",
                    "    colors = {1: 'red', 2: 'blue', 4: 'green'} # 1: vehicle, 2: pedestrian, 4: cyclist\n",
                    "    \n",
                    "    for idx in range(num_images):\n",
                    "        ax = axes[idx]\n",
                    "        ax.imshow(images[idx])\n",
                    "        h, w, _ = images[idx].shape\n",
                    "        \n",
                    "        # Draw boxes\n",
                    "        for box, cls in zip(boxes[idx], classes[idx]):\n",
                    "            if cls == 0: continue # Skip padded elements\n",
                    "            ymin, xmin, ymax, xmax = box\n",
                    "            color = colors.get(cls, 'yellow')\n",
                    "            rect = patches.Rectangle((xmin * w, ymin * h), (xmax - xmin) * w, (ymax - ymin) * h,\n",
                    "                                     linewidth=2, edgecolor=color, facecolor='none')\n",
                    "            ax.add_patch(rect)\n",
                    "        ax.axis('off')\n",
                    "        ax.set_title(f'Sample {idx+1}')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Load dataset and visualize"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Update the path below to map to your data folder\n",
                    "tfrecord_files = glob.glob('../data/train/*.tfrecord')\n",
                    "if tfrecord_files:\n",
                    "    dataset = get_dataset(tfrecord_files[0]) # Get the first dataset\n",
                    "    batch = next(iter(dataset)) # iterate\n",
                    "    display_instances(batch, max_images=3)\n",
                    "else:\n",
                    "    print('No TFRecords found. Please generate them first.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Data Distribution Analysis\n",
                    "We need to understand class imbalance. Let's count the occurrences of each class over a few tfrecords."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from collections import Counter\n",
                    "class_counts = Counter()\n",
                    "\n",
                    "if tfrecord_files:\n",
                    "    print(\"Analyzing data distribution...\")\n",
                    "    # Limit to 5 files to avoid excessive time during EDA\n",
                    "    for f in tfrecord_files[:5]:\n",
                    "        ds = get_dataset(f)\n",
                    "        for elem in ds:\n",
                    "            classes = elem['groundtruth_classes'].numpy().flatten()\n",
                    "            class_counts.update(classes[classes > 0]) # mask out zeros\n",
                    "            \n",
                    "    labels = [\"Vehicle (1)\", \"Pedestrian (2)\", \"Cyclist (4)\"]\n",
                    "    counts = [class_counts.get(1, 0), class_counts.get(2, 0), class_counts.get(4, 0)]\n",
                    "    \n",
                    "    plt.figure(figsize=(8,5))\n",
                    "    plt.bar(labels, counts, color=['red', 'blue', 'green'])\n",
                    "    plt.title('Object Class Distribution in Sample')\n",
                    "    plt.ylabel('Count')\n",
                    "    plt.show()\n",
                    "    print(f\"Raw Class Counts: {class_counts}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(out_path, "w") as f:
        json.dump(eda_nb, f, indent=2)
    print(f"Created {out_path}")

def create_inference_notebook(out_path):
    inference_nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Inference and Video Animation\n",
                    "This notebook demonstrates how to load the exported trained model, pass a video or TFRecord dataset sequence into it, and generate an animated video of the bounding box detections."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "sys.path.append('../scripts')\n",
                    "from inference_video import main as build_video\n",
                    "\n",
                    "# Define Paths\n",
                    "LABELMAP_PATH = '../label_map.pbtxt'\n",
                    "MODEL_PATH = '../experiments/pretrained_model/saved_model' # Replace with exported model path\n",
                    "TF_RECORD_PATH = '../data/test/segment-12345.tfrecord' # Replace with your test tfrecord\n",
                    "CONFIG_PATH = '../experiments/pretrained_model/pipeline.config'\n",
                    "OUTPUT_PATH = 'animation.mp4'\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Generate Animation\n",
                    "We execute the exported `detect_fn` dynamically on frames of the TFRecord, visualizing the labeled boxes using `matplotlib.animation`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# If the files exist, this will build the mp4. \n",
                    "# Note: running this requires standard TF2 environment and matplotlib.\n",
                    "if os.path.exists(MODEL_PATH) and os.path.exists(TF_RECORD_PATH):\n",
                    "    print('Starting Video Generation...')\n",
                    "    build_video(LABELMAP_PATH, MODEL_PATH, TF_RECORD_PATH, CONFIG_PATH, OUTPUT_PATH)\n",
                    "    print('Video saved to animation.mp4!')\n",
                    "else:\n",
                    "    print('Model or Testing data missing. Ensure you have exported the model first.')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(out_path, "w") as f:
        json.dump(inference_nb, f, indent=2)
    print(f"Created {out_path}")

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    create_eda_notebook("notebooks/Exploratory_Data_Analysis.ipynb")
    create_inference_notebook("notebooks/Inference_and_Animation.ipynb")
