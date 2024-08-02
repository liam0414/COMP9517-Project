# COMP9517 2024 T2 Group Project README

This paper explores natural scene semantic segmentation using the WildScenes dataset. We develop and test three baseline models: U-Net, SegNet and YOLOv8. We implement various data-preprocessing such as mean subtraction, hyperparameter tuning and post-processing methods involving superpixels, conditional random fields (CRF), pixel and morphological operations, and ensemble learning to improve our results. Based on our results, we discuss challenges in natural scene segmentation and conclude by proposing future work avenues.

![image](https://github.com/user-attachments/assets/43572984-2aec-409d-8c36-d557afb1de49)


## Folder Structure
Our code submission is organized as follows:

- **Root Folder:**
  - `README.md`: This file.
  - `eda.ipynb`: Contains all the code for performing exploratory data analysis.
  - `model_run.ipynb`: Contains the code for running our final models. Running this notebook from start to finish, after setting up your environment according to the requirements listed below, will reproduce the demonstrated results.

- **Subfolders:**
  - `models`: Contains the code for our final models and final post-processing code.
  - `notebooks`: Contains all our progress throughout the project, including models we decided not to include as our final models such as SegNet and U-Net. It also contains code for our post-processing techniques such as superpixels, morphology, and CRF.

## Package Requirements

To set up the environment, run the following commands:

```bash
pip install 
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

### Installing CRF
The package to run CRF algorithm requires the following module set-up in one of two ways.
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

### Running the Code
- Set Up Environment:
  - Ensure you have Python 3.8.19 installed.
  - Install all the required packages listed above.
- Exploratory Data Analysis:
  - Open and run eda.ipynb to perform exploratory data analysis.
- Model Training and Evaluation:
  - Open and run model_run.ipynb to train and evaluate our final models. This notebook will guide you through the steps to reproduce the results demonstrated in our report.

### Contact Information
For any questions or issues, please contact the team members:
* Liam Chen: z3278107@ad.unsw.edu.au
* The Duy Nguyen: z5532839@ad.unsw.edu.au
* Shree Baskar: z5444481@ad.unsw.edu.au
* Jennifer Wang: z5311246@ad.unsw.edu.au
* Amulya Lokeshwari Sri Raja Kalidindi: z5520321@ad.unsw.edu.au
