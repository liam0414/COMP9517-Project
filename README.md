# COMP9517 2024 T2 Group Project README

This paper explores natural scene semantic segmentation using the WildScenes dataset. We develop and test three baseline models: U-Net, SegNet and YOLOv8. We implement various data-preprocessing such as mean subtraction, hyperparameter tuning and post-processing methods involving superpixels, conditional random fields (CRF), pixel and morphological operations, and ensemble learning to improve our results. Based on our results, we discuss challenges in natural scene segmentation and conclude by proposing future work avenues.

![image](https://github.com/user-attachments/assets/43572984-2aec-409d-8c36-d557afb1de49)


# Folder Structure
Our code submission is organized as follows:

- **Top level files:**
  - `README.md`: This file.
  - `eda.ipynb`: Contains all the code for performing exploratory data analysis.
  - `model_run.ipynb`: Contains the code for running our final models. Running this notebook from start to finish, after setting up your environment according to the requirements listed below, will reproduce the demonstrated results.
  * NOTE: Top level notebooks are all that is needed to demonstrate the results.

- **Subfolders:**
  - `models`: Contains the code for our final models and final post-processing code.
  - `notebooks`: Contains all our progress throughout the project, including certain highlighted models we decided not to include as our final models such as SegNet and U-Net. It also contains code for our post-processing techniques such as superpixels, morphology, and CRF. These notebooks are also all runnable.
  - `dataset`: Contains `.csv` for the original, and custom test and validation samples (splitting mechanism outline in the accompanying report), and 30% stratified samples.

**NOTE**: The `WildScenes2d` dataset folderr should be at top level. The following should provide comprehensive guide to the folder structure:
```python
COMP9517-main/
│
├── dataset/
│   ├── test.csv             # original test
│   ├── train.csv            # original train
│   ├── val.csv              # original val
│   ├── train_sample_30.csv  # 30% train
│   ├── val_sample_30.csv    # 30% val
│   └── train_sample.csv     # 50% train
│
├── models/
│   ├── unetpp/ # model type
│   │   ├──     # backbone .py files for model
│   │  ...
│   │
│  ...
│
├── notebooks/
│   ├── unetpp/ # model type
│   │   ├──     # notebooks for detailing train and evaluation
│   │  ...
│   │
│  ...
│
├── WildScenes2d/ # wildscenes dataset
│   ├── K-01
│   ├── K-03
│   ├── V-01
│   ├── V-02
│   └── V-03
│
├── eda.ipynb       # final EDA results
└── model_run.ipynb # final prediction results
```
    
# Package Requirements

To set up the environment, run the following commands:

```bash
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install opencv-python
pip install pandas
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install scikit-image
pip install cython
pip install matplotlib
pip install tqdm
```

**Note**: we ran our code using a Nvidia GPU on Windows platform, ensure your system meets the requirements for CUDA installation by checking the [CUDA Toolkit Documentation]([url](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)). Install the NVIDIA driver for your GPU. You can find the appropriate driver [here]([url](https://www.nvidia.com/Download/index.aspx)).

to verify if you have cuda
```
nvcc --version
```

## Installing CRF
The package to run CRF algorithm requires the following module set-up in one of two ways. (Please note that CRF is not used in our final results, so this could be optional.)
1. (via pip): 
```bash
pip install pydensecrf
```
2. (via git):
```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

This repository is a Cython-based Python wrapper for Philipp Krähenbühl's Fully-Connected CRFs. The repository notes the requirement of a recent version of Cython (at least version 0.22) to run the code, which can be installed via pip
```bash
pip install cython
```

## Running the Code
- Set Up Environment:
  - Ensure you have Python 3.11+ installed.
  - Install all the required packages listed above.
- Exploratory Data Analysis:
  - Open and run `eda.ipynb` to perform exploratory data analysis.
- Model Evaluation:
  - Open and run `model_run.ipynb` to evaluate our final models. This notebook will guide you through the steps to reproduce the results demonstrated in our report.
  - To run model training and other training evaluations, go to `notebooks/` folder, and run any notebook in its subfolders.
