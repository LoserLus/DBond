# Dbond
Implementation for paper Optimizing Mirror-Image Peptide Sequence Design for Data Storage via Peptide Bond Cleavage Prediction
# Build
use dockerfile to build docker image
```bash
docker build -t dbond_env:v0 .
```

# Files
```bash
DBond/
|-- LICENSE
|-- README.md
|-- best_model              # Stores best-performing model weights
|   |-- dbond_m
|   `-- dbond_s
|-- checkpoint              # Periodic model weight checkpoints
|   |-- dbond_m
|   `-- dbond_s
|-- data_utils_dbond_m.py   # Data utilities for dbond-m
|-- data_utils_dbond_s.py   # Data utilities for dbond-s
|-- dataset                 # Dataset directory
|   |-- dataset.fbr.csv     # Example dataset used for evaluate dbond-s 
|   |-- dbond_m.test.csv    # Example test set for dbond-m
|   |-- dbond_m.train.shuffle.csv   
|   |-- dbond_s.test.csv    # Example test set for dbond-s
|   `-- dbond_s.train.shuffle.csv
|-- dbond_m.py              # Model architecture definition for dbond-m
|-- dbond_m_config          # Model configuration files for dbond-m
|   `-- default.yaml
|-- dbond_s.py              # Model architecture definition for dbond-s
|-- dbond_s_config          # Model configuration files for dbond-s
|   `-- default.yaml
|-- dockerfile              # Dockerfile for building the Docker image
|-- evaluate.dbond_m.py     # Evaluate multi-label metrics on dbond-m test set
|-- evaluate.dbond_s.py     # Evaluate multi-label metrics on dbond-s test set
|-- multi_label_metrics.py  # Utility functions for computing multi-label metrics
|-- result                  # Stores evaluation results
|   |-- dbond_m
|   |-- dbond_s
|   `-- multi_label_metric  # Stores multi-label metrics result files
|-- tensorboard             # TensorBoard logs
|   |-- dbond_m
|   `-- dbond_s
|-- train.dbond_m.py        # Training script for dbond-m
`-- train.dbond_s.py        # Training script for dbond-s
```