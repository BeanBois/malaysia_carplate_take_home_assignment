# Malaysian License Plate Recognition System

An end-to-end deep learning solution for detecting and recognizing Malaysian license plates in images and videos.

## Requirements

- Currently only tested on MacOS, and should work fine on Linux. I am unsure about windows though due to different file seperators (untested)

## Quick Start

### 1. Installation

```bash
# Clone or extract the repository
git clone git@github.com:BeanBois/malaysia_carplate_take_home_assignment.git
cd malaysian_lpr

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

#### Installing pre-Trained YOLOv8 
install the weights [here](https://drive.google.com/file/d/1m0z9lXV3Fiwxj30PduxMuzsZsTEMuDI5/view?usp=share_link) and name it 'yolov8n.pt' in current working directory

### 2. Data Preparation

#### Training Datasets 
Download dataset [here](https://drive.google.com/drive/folders/1DiJLBEiLDUvSlWKiaGVzHwWQe-DBbtRF?usp=share_link) and place under data/ (link should redirect you to 3 folders: "raw", "additional_data" and "negative_examples")

#### Testing Datasets
Download dataset [here](https://drive.google.com/drive/folders/19NsJkiZWOFiFuSynyAfUvp2IVP_rtAgd?usp=share_link) and place under testdata/ (link should redirect you to a folder called 'images'. put this 'images' folder under testdata/)

### 3. Training

```bash
 python train.py --mode integrate --augment    
```

### 4. Inference and Evaluation

```bash
python inference.py --directory testdata/images/ --model models/detection_training/plate_detector{n}/weights/best.pt --multi-engine
```
*n depends on how many times you have trained the model. Basically locate the desired model under models/detection_training/plate_detector{n}/weights/best.pt. Usually, this will be 'plate_detector/' on first training, 'plate_detector2' on 2nd training and so on ...*



## 📁 Project Structure (impt stuff only)

```
malaysian_lpr/
├── config.yaml                 # Configuration file
├── requirements.txt           # Python dependencies
├── train.py                   # Training script
├── inference.py               # Inference and Evaluation script
│
├── src/
│   ├── detector.py           # Detection module (YOLOv8)
│   ├── recognizer.py         # Recognition module (OCR)
│   ├── pipeline.py           # End-to-end pipeline
│   ├── data_preparation.py   # Data augmentation
│   ├── data_preparation.py   # Data augmentation
│   ├── integrate_additional_data.py   # Data util to add additional non-Malaysian plate carplates
│   ├── augment_integrated_data.py   # Data augmentation for integrated data
│   └── add_negative_examples.py    # Data util to add negative examples 
│
├── data/
│   ├── raw/                  # Original dataset (negatives included, use 'add_negative_examples.py' to remove/add them back)
│   │    ├── test/                 # test dataset
│   │    ├── val/                  # validation dataset
│   │    └── train/                # training dataset (val and test follow the same format)
│   │         ├── images/                # images 
│   │         └── labels/                # labels
│   ├── additional_data/      # Additional dataset (non-Malaysian)
│   │    ├── images/               # additional images
│   │    └── annotations/          # additional annotations
│   └── negative_examples/    # Negative examples (Roadsign ect.)
│
├── testdata/                 # for evaluation and inference (not the same as those under data!)
│   └──  images/                  # test images (names are labels)
├── models/                    # Trained models
│    └── detection_training/      
│        ├── plate_detector/       # first training
│        └── plate_dectector2/     # second training (contains results plots for detector training and weights)
│            ├── results.png           # results plots
│            └── weights/              # weights
│                ├── best.pt/            # best performance
│                └── last.pt/            # last epoch
│            ...
└── outputs/                   # Results and visualizations
     └── results/                  # inference and evaluation results and visualisation
         ├── summary.json    # summary of metrics 
         ├── detailed_results.json  # results from inference
         ├── *.jpg           # detected (cropped image)
         └── *_results.jpg   # recognised (with confidence score)
```
