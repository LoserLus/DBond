# Dbond

This project implements the methods described in the paper Optimizing Mirror-Image Peptide Sequence Design for Data Storage via Peptide Bond Cleavage Prediction. Its goal is to optimize mirror-image peptide sequence design through peptide bond cleavage prediction.

## Project Structure

```
.
├── Dockerfile                # Docker environment configuration
├── LICENSE                   # License
├── PBCLA/                    # PBCLA tools and related scripts
│   ├── mgf2csv.dbond_m.py
│   ├── mgf2csv.dbond_s.py
│   ├── mgf_dataset/
│   ├── pbcla.py
│   └── utils.py
├── best_model/               # Best model weights
│   ├── dbond_m/
│   └── dbond_s/
├── checkpoint/               # Training checkpoints
│   ├── dbond_m/
│   └── dbond_s/
├── data_utils_dbond_m.py     # Data processing utilities for dbond_m
├── data_utils_dbond_s.py     # Data processing utilities for dbond_s
├── dataset/                  # Dataset files
│   ├── *.csv
├── dbond_m.py                # Main model definition for dbond_m
├── dbond_m_config/           # Configuration for dbond_m
│   └── default.yaml
├── dbond_s.py                # Main model definition for dbond_s
├── dbond_s_config/           # Configuration for dbond_s
│   ├── 1222_h_256_adam_4_b4_dn.yaml
│   └── default.yaml
├── evaluate.dbond_m.py       # Evaluation script for dbond_m
├── evaluate.dbond_s.py       # Evaluation script for dbond_s
├── multi_label_metrics.py    # Multi-label evaluation metrics
├── result/                   # Output results
│   ├── metric/
│   └── pred/
├── tensorboard/              # TensorBoard logs
│   ├── dbond_m/
│   └── dbond_s/
├── train.dbond_m.py          # Training script for dbond_m
├── train.dbond_s.py          # Training script for dbond_s
└── README.md                 # Project documentation
```

## Main Files

- `dbond_m.py` / `dbond_s.py`: model architecture definitions
- `train.dbond_m.py` / `train.dbond_s.py`: training workflows
- `evaluate.dbond_m.py` / `evaluate.dbond_s.py`: evaluation workflows
- `multi_label_metrics.py`: multi-label classification metrics
- `PBCLA/`: peptide bond cleavage analysis tools

## Quick Start

### 1. Environment

A prebuilt runtime image is available on Docker Hub and is the recommended way to run the project:

```bash
docker pull LoserLus/dbond_env:latest
```

You can also build the image locally:

```bash
docker build -t dbond_env:v0 .
```

### 2. Data Preparation

Place the dataset files in the `dataset/` directory. Refer to the existing CSV files for the expected format.

The current training pipeline uses explicit `train`, `validation`, and `test` splits.

### 3. Train the Models

Example for dbond_m:

```bash
python train.dbond_m.py --config dbond_m_config/default.yaml
```

Example for dbond_s:

```bash
python train.dbond_s.py --config dbond_s_config/default.yaml
```

### 4. Evaluate the Models

#### Evaluate dbond_m

```bash
python evaluate.dbond_m.py \
	--in_model_weight_path best_model/dbond_m/2025_11_25_19_25_default_0.pt \
	--in_model_comfig_path dbond_m_config/default.yaml \
	--in_csv_to_predict_path dataset/dbond_m.test.csv \
	--out_multi_label_pred_dir result/pred/dbond_m/multi/ \
	--out_multi_label_metric_dir result/metric/dbond_m/multi/
```

#### Evaluate dbond_s

```bash
python evaluate.dbond_s.py \
	--in_model_weight_path best_model/dbond_s/2025_11_29_18_46_default_5.pt \
	--in_model_comfig_path dbond_s_config/default.yaml \
	--in_csv_to_predict_path dataset/dbond_s.test.csv \
	--in_csv_for_multi_label_path dataset/dataset.fbr.csv \
	--out_single_label_pred_dir result/pred/dbond_s/single/ \
	--out_multi_label_pred_dir result/pred/dbond_s/multi/ \
	--out_single_label_metric_dir result/metric/dbond_s/single/ \
	--out_multi_label_metric_dir result/metric/dbond_s/multi/
```

#### Argument Reference

- `--in_model_weight_path`: path to the model weight file
- `--in_model_comfig_path`: path to the model configuration file in YAML format
- `--in_csv_to_predict_path`: path to the input CSV file for prediction
- `--in_csv_for_multi_label_path` (dbond_s only): path to the CSV file used for multi-label evaluation
- `--out_single_label_pred_dir` (dbond_s only): output directory for single-label predictions
- `--out_multi_label_pred_dir`: output directory for multi-label predictions
- `--out_single_label_metric_dir` (dbond_s only): output directory for single-label metrics
- `--out_multi_label_metric_dir`: output directory for multi-label metrics

The evaluation scripts automatically save prediction results and metrics to the specified directories.

### 5. PBCLA Tools

The `PBCLA/` directory contains peptide bond cleavage utilities for data conversion and analysis.

For more details about the data processing workflow and field definitions, see [PBCLA/Readme.md](/Users/luyilong/Downloads/DBond/PBCLA/Readme.md).