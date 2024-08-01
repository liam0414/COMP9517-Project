# COMP9517 2024 T2
# Group Project README

## Folder Structure
Our code submission is organised as follows. In the root folder, we have this README.md file and two notebooks: eda.ipynb and model_run.ipynb.
eda.ipynb contains all the code we created to perform exploratory data analysis.
model_run.ipynb contains our final models results. Running the notebook from start to finish, after setting up your environment according to the requirements listed below, will produce the demonstrated results.
There are two subfolders: models and notebooks. Notebooks contains all our progress for the entire project, including models we decided not to include as our final models such as SegNet and U-Net, and our final models (U-Net Plus Plus and YOLOv8). It also contains code for our postprocessing techniques such as superpixels, morphology, and CRF.
The models folder contains just the code for our final models and final post-processing code.


## Package Requirements
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install opencv-python
pip install pandas

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install ultralytics
pip install scikit-image
pip install cython

### Installing CRF
The package to run CRF algorithm requires the following module set-up in one of two ways.

1. (via pip): pip install pydensecrf
2. (via git): pip install git+https://github.com/lucasb-eyer/pydensecrf.git

This repository is a Cython-based Python wrapper for Philipp Krähenbühl's Fully-Connected CRFs. The repository notes the requirements of a recent version of Cython (at leaset version 0.22) to run the code which can be installed via pip (pip install cython).
