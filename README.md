# VITAL
VITAL is a deep learning framework that co-learns local structural geometries and global sequence contexts to enable quantitative peptide–protein interaction (PepPI) characterization.

![Model architecture](picture/model_arch.tif)

<details open><summary><b>Table of contents</b></summary>

- [Code organization](#code-org)
- [Requirement](#require)
- [Usage](#usage)
    - [Installation](#install)
    - [Feature Extraction](#feature)
    - [Inference](#inference)
- [Web server](#server)
</details>

### Code organization <a name="code-org"></a>
* `ckpts/` - Pretrained model checkpoints
* `datasets/` - Example datasets used for training and evaluation?
* `data_processing/` - Tools and utilities for feature extraction and preprocessing
* `model/` - Contains the model file for inference
* `feature_dic.py` - Main script for generating feature dictionariesrun_feature.sh
* `parse_feature_dict.py` - Script for parsing and organizing extracted feature dictionaries
* `prediction.py` - Main script for running inference
* `run_feature.sh` - Shell script for executing the full feature extraction pipeline
* `run_prediction.sh` - Shell script for performing model prediction

### Requirement <a name="require"></a>
All experiments were conducted using PyTorch 1.12.1 and Python 3.9 on a server equipped with an NVIDIA GeForce RTX 3090 GPU (CUDA 11.4).

### Usage <a name="usage"></a>
#### Installation <a name="install"></a>
Follow the steps below to set up the environment and install all dependencies.

**1. Clone the repository**

```text
git clone https://github.com/BADD-XMU/VITAL.git
cd VITAL
```

**2. Create the Conda environment**

We provide an `env.yml` file for reproducible environment setup.

```text
conda env create -f env.yml
conda activate VITAL
```

#### Feature Extraction <a name="feature"></a>
Before using VITAL for inference, you need to generate all required features.  
Run the full feature extraction pipeline:

```text
bash run_feature.sh
```

Or run it manually if you wish to modify default arguments via command line:

```text
python feature_dic.py \
    --load_list ./datasets/example_data/example_list \
    --load_fasta ./datasets/example_data/example_fasta/ \
    --save_path ./datasets/example_feature/
```

The script will:
* Generate sequence-based and structure-related features
* Save processed feature dictionaries into `./datasets/example_feature/`

#### Inference <a name="inference"></a>
Run the full inference pipeline:

```text
bash run_prediction.sh
```

Or run inference manually:

```text
python prediction.py \
    --batch_input_csv ./datasets/example_feature/feature_path.csv \
    --ckpt_path ./ckpts/VITAL_PePPI_checkpoint.pkl \
    --device cuda:0 \
    --output ./output/prediction_results/result.json \
    --ASM_output_path ./output/ASM \
    --verbose
```

The inference script will:
* Load the precomputed feature dictionary
* Load the TorchScript model
* Produce prediction scores and ASM for each protein–peptide pair
* Inference results will be saved to prediction_results.csv by default.

### Web server <a name="server"></a>
You can access and use VITAL through the [VITAL-web-server](https://www.vital-peppi.online/).



