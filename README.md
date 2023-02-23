# Multiple Instance Probability Contrastive Learning (MIPCL)

Code repository for paper: "Deep learning-based screening and model explainability for pancreatic cancer cytology".

Please refer to paper for methods.

# Table of Contents

1. [Pre-requisites](#pre-requisites)
2. [Install](#Install)
3. [Data](#data)
4. [Usage](#usage)

# Pre-requisites

* Linux (Tested on Ubuntu 20.04)
* NVIDIA GPU (Tested on up to x4 NVIDIA Tesla V100s on local server)
  * Both (1) preprocessing and (2) model training require at least x1 GPU. For our study, we used x4 and x1 GPU(s) for (1) and (2), respectively
* Python (3.8.13), NumPy (1.23.3), OpenCV-Python (4.6.0), pyvips (2.2.1), scikit-learn (0.19.3), pandas (1.4.4), h5py (3.7.0), Matplotlib (3.5.2), PyTorch (1.12.1), Torchvision (0.13.1), PyTorch-Lightning (1.7.7), torchmetrics (0.11.1), timm (0.6.12), tensorboard (2.10.1), smooth-topk (1.0)

# Installation Guide

For instructions on installing anaconda on your machine (download the distribution that comes with python 3): (<https://www.anaconda.com/distribution/>)

After setting up anaconda, use the environment configuration file **mipcl.yaml** to create a conda environment:

```sh
conda env create -n mipcl -f ./clam.yaml
```

Activate the create environment:

```sh
conda activate mipcl
```

Clone our codebase:

```sh
git clone https://github.com/ansohn88/MIPCL.git
cd MIPCL
```

Once inside codebase, to install smooth-topk for CLAM:

```sh
git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
python setup.py install
```

When done running experiments, deactivate environment:

```sh
conda deactivate mipcl
```

# Data

==TODO==: Instructions to access data on Proscia will be included.

# Usage

The steps required to reproduce the results of the paper are:

1. Tiling the whole slide images
2. Extracting features from the tiles using a pretrained network
3. Training and testing the MIPCL model
4. Visualization

### Tessallating the whole slide images

This step will generate the tessallated tiles from the whole slide images to specified output directory in h5py format. The amount of time it takes per whole slide image depends on the foreground segmentation, but should be under 1 minute max.

```sh
python ./preprocessing/tiler.py --wsi_dir DATA_DIRECTORY --out_pdir OUTPUT_DIRECTORY --csv FNAME_LBL_CSV --z Z_LEVEL --n_jobs NUM_CORES
```

---------------------------------------

### Extract features with a pretrained network

This step will generate the extracted features with a pretrained network (ConvNeXt used in paper), and saved in h5py format to specified output directory. The amount of time it takes per case depends on how many tiles were extracted from the tessallation step above, but should be under 1 minute max.

```sh
python ./preprocessing/feats_extract.py --tiles_dir TILES_DIR --tile_size TILE_SIZE --out_pdir OUTPUT_DIRECTORY --class_path FNAME_LBL_CSV --model_name MODEL_NAME --z_stack USE_ALL_Z --z_level Z_LEVEL --device_ids GPU_IDS
```

---------------------------------------------

### Training and Evaluating Model

##### MIPCL with InfoNCE instance loss

Results (final logits, final predictions, metric values, top tile indices) across the ten folds will be saved as a pickle file to the specified output directory. Training, evaluating and testing each fold depends on both the `--model` and `--patience` flags, but should take somewhere between 50-90 minutes.

```sh
python trainer.py --data_root_dir PRE_FEATS_DIR --kfold_splits_csv_dir CSV_SPLITS_FILE --results_dir RESULTS_DIR --model MODEL --bag_weight BAG_WT --in_channels INPUT_DIM --intermediate_dim INTER_DIM --stain_info USE_STAIN_INFO --drop USE_DROPOUT --mipcl_temp TEMP --mipcl_thresh P_THRESH
```

##### MIPCL with Cosine Similarity instance loss

```sh
python trainer.py --data_root_dir PRE_FEATS_DIR --kfold_splits_csv_dir CSV_SPLITS_FILE --results_dir RESULTS_DIR --model MODEL --bag_weight BAG_WT --in_channels INPUT_DIM --intermediate_dim INTER_DIM --stain_info USE_STAIN_INFO --drop USE_DROPOUT --mipcl_alpha CS_ALPHA
```

##### CLAM

```sh
python trainer.py --data_root_dir PRE_FEATS_DIR --kfold_splits_csv_dir CSV_SPLITS_FILE --results_dir RESULTS_DIR --model MODEL --bag_weight BAG_WT --in_channels INPUT_DIM --intermediate_dim INTER_DIM --stain_info USE_STAIN_INFO --drop USE_DROPOUT --clam_topk TOP_K --clam_inst_loss SVM_OR_CE
```

##### ABMIL

```sh
python trainer.py --data_root_dir PRE_FEATS_DIR --kfold_splits_csv_dir CSV_SPLITS_FILE --results_dir RESULTS_DIR --model MODEL --bag_weight BAG_WT --in_channels INPUT_DIM --intermediate_dim INTER_DIM --stain_info USE_STAIN_INFO --drop USE_DROPOUT
```

------------

### Visualization

This step can generate the top tiles, top tiles index, top derived probabilities, and all predictions with labels:

```sh
python vis_topk_tiles.py --tiles_dir TILES_DIRECTORY --output_dir OUTPUT_DIRECTORY --model_results MODEL_FOLD_RESULTS_PATH --fold_num FOLD_NUM --get_all_preds RETURN_FOLD_PREDS --get_top_ids RETURN_FOLD_TILE_IDS --get_top_probs RETURN_FOLD_TOP_PROBS --which_metric RETRIEVE_METRIC
```
