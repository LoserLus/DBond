# Dbond

本项目实现了论文《Optimizing Mirror-Image Peptide Sequence Design for Data Storage via Peptide Bond Cleavage Prediction》中的相关方法，旨在通过肽键断裂预测优化镜像肽序列设计

## 项目结构

```
.
├── Dockerfile                # Docker 环境配置
├── LICENSE                   # 许可证
├── PBCLA/                    # PBCLA 工具及相关脚本
│   ├── mgf2csv.dbond_m.py
│   ├── mgf2csv.dbond_s.py
│   ├── mgf_dataset/
│   ├── pbcla.py
│   └── utils.py
├── best_model/               # 最优模型权重
│   ├── dbond_m/
│   └── dbond_s/
├── checkpoint/               # 训练过程中的模型检查点
│   ├── dbond_m/
│   └── dbond_s/
├── data_utils_dbond_m.py     # 数据处理工具（m 型）
├── data_utils_dbond_s.py     # 数据处理工具（s 型）
├── dataset/                  # 数据集文件
│   ├── *.csv
├── dbond_m.py                # m 型主模型脚本
├── dbond_m_config/           # m 型模型配置
│   └── default.yaml
├── dbond_s.py                # s 型主模型脚本
├── dbond_s_config/           # s 型模型配置
│   ├── 1222_h_256_adam_4_b4_dn.yaml
│   └── default.yaml
├── evaluate.dbond_m.py       # m 型模型评估脚本
├── evaluate.dbond_s.py       # s 型模型评估脚本
├── multi_label_metrics.py    # 多标签评估指标
├── result/                   # 结果输出
│   ├── metric/
│   └── pred/
├── tensorboard/              # TensorBoard 日志
│   ├── dbond_m/
│   └── dbond_s/
├── train.dbond_m.py          # m 型模型训练脚本
├── train.dbond_s.py          # s 型模型训练脚本
└── README.md                 # 项目说明
```

## 主要文件说明

- `dbond_m.py` / `dbond_s.py`：主模型结构定义
- `train.dbond_m.py` / `train.dbond_s.py`：训练流程
- `evaluate.dbond_m.py` / `evaluate.dbond_s.py`：评估流程
- `multi_label_metrics.py`：多标签分类评估指标
- `PBCLA/`：肽键断裂分析工具

## 快速开始



### 1.运行环境

已在 Docker Hub 上传预构建的运行环境镜像，建议直接拉取使用：

```bash
docker pull LoserLus/dbond_env:latest
```

也可自行构建本地镜像：

```bash
docker build -t dbond_env:v0 .
```

### 2. 数据准备

将原始数据集放入 `dataset/` 目录，格式参考已有 csv 文件

### 3. 训练模型

以 m 型模型为例：

```bash
python train.dbond_m.py --config dbond_m_config/default.yaml
```

s 型模型同理：

```bash
python train.dbond_s.py --config dbond_s_config/default.yaml
```


### 4. 评估模型

#### dbond_m 评估

```bash
python evaluate.dbond_m.py \
	--in_model_weight_path best_model/dbond_m/2025_11_25_19_25_default_0.pt \
	--in_model_comfig_path dbond_m_config/default.yaml \
	--in_csv_to_predict_path dataset/dbond_m.test.csv \
	--out_multi_label_pred_dir result/pred/dbond_m/multi/ \
	--out_multi_label_metric_dir result/metric/dbond_m/multi/
```

#### dbond_s 评估

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

#### 参数说明

- `--in_model_weight_path`：模型权重文件路径
- `--in_model_comfig_path`：模型配置文件路径（yaml）
- `--in_csv_to_predict_path`：待预测的 csv 文件路径
- `--in_csv_for_multi_label_path`（仅 dbond_s）：用于多标签评估的 csv 文件路径
- `--out_single_label_pred_dir`（仅 dbond_s）：单标签预测结果保存目录
- `--out_multi_label_pred_dir`：多标签预测结果保存目录
- `--out_single_label_metric_dir`（仅 dbond_s）：单标签评估结果保存目录
- `--out_multi_label_metric_dir`：多标签评估结果保存目录

评估脚本会自动保存预测结果和评估指标到指定目录


### 5. PBCLA 工具

PBCLA 目录下包含肽键断裂相关工具，可用于数据转换和分析

**数据处理流程、字段解释等详细说明请参见 [`PBCLA`](PBCLA/Readme.md)**

### 6 预训练模型权重与配置

训练好的模型权重和对应的配置文件已上传至 Hugging Face：

- 地址：https://huggingface.co/LoserLus/Dbond

可直接下载并用于评估或推理