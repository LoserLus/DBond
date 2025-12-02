
# PBCLA 目录说明

PBCLA 目录包含肽键断裂分析相关工具，主要用于 MGF 文件处理、多肽碎片键标注及数据格式转换


## 文件功能简介

- `pbcla.py`：
  - PBCLA算法核心实现，输入 MGF 文件，计算多肽碎片键数、缺失碎片键等信息，并将结果写回 MGF 文件
  - 支持命令行参数：
    ```bash
    python pbcla.py --in_mgf_path ./mgf_dataset/example.mgf --out_mgf_path ./mgf_dataset/example_out.mgf
    ```

- `mgf2csv.dbond_m.py`：
  - 将 PBCLA 处理后的 MGF 文件转换为 dbond_m 训练/评估所需的 CSV 格式，包含多标签信息
  - 支持命令行参数：
    ```bash
    python mgf2csv.dbond_m.py --mgf_path ./mgf_dataset/example_out.mgf --csv_path ./mgf_dataset/example.multi.csv
    ```

- `mgf2csv.dbond_s.py`：
  - 将 PBCLA 处理后的 MGF 文件转换为 dbond_s 训练/评估所需的 CSV 格式，包含单标签和多标签信息
  - 支持命令行参数：
    ```bash
    python mgf2csv.dbond_s.py --mgf_path ./mgf_dataset/example_out.mgf --csv_path ./mgf_dataset/example.csv
    ```

- `utils.py`：
  - 提供多肽对象、质量阈值计算、二分查找等 PBCLA 算法辅助工具

- `mgf_dataset/`：
  - 存放示例 MGF 文件及转换结果，如 `example.mgf`、`example_out.mgf`、`example.csv`、`example.multi.csv`


## CSV 字段说明（多标签分类任务背景）

PBCLA 处理后的 CSV 文件用于多标签分类任务，输入输出字段如下：

### 输入字段

- `seq`：镜像蛋白的氨基酸序列，字符串类型。例如 `AEFDE` 表示由 6 个氨基酸构成的序列，共包含 5 个肽键
- `charge`：母离子的带电量，整型。例如 `2` 表示带 2 个正电荷
- `pep_mass` 或 `m_z`：母离子的质荷比，浮点型。例如 `544.6621`
- `intensity`：信号强度，浮点型。例如 `116622.119`
- `nce`：碰撞能量强度，整型。例如 `30` 表示碰撞能量为 30ev
- `scan_num`：谱图扫描次序，整型。例如 `219` 表示第 219 次扫描
- `rt`：保留时间（Retention Time），浮点型
- `bond_aa`（dbond_s）：肽键对应的二肽氨基酸片段，如 `AE`
- `bond_pos`（dbond_s）：肽键在序列中的位置，从 0 开始

### 输出字段

- `bond_label`（dbond_s）：单个肽键断裂情况，0 表示未断裂，1 表示断裂
- `true_multi`（dbond_m）：多标签断裂序列，形如 `1;0;1;0;1`，每个位置代表一个肽键断裂情况
- `mb`：缺失的碎片键位置，分号分隔，如 `1;3;5;` 表示第 1、3、5 个肽键缺失
- `tb`：所有的肽键的数量
- `fb`：可以匹配到的肽键（断裂的肽键）的数量
- `fbr`：fragmented bond ratio，fbr = fb/tb，表示肽键断裂的比例

#### 输出解释
对于序列 `AEFDE`，其断裂序列如 `[1,0,1,0,1]`，表示：
- 第 1 个肽键（AE）断裂
- 第 2 个肽键（EF）未断裂
- 第 3 个肽键（FD）断裂
- 第 4 个肽键（DE）未断裂
- 第 5 个肽键（ED）断裂


## 数据处理链路

一个基本的数据处理流程如下：

1. **原始串联质谱数据转换为 MGF 格式**
  - 例如将 RAW 格式数据通过 MSConvert 工具转换为 MGF 文件
2. **MGF 文件标注**
  - 使用 `pbcla.py` 对原始 MGF 文件进行肽键断裂标注，生成带有碎片信息的 MGF 文件
3. **MGF 转换为 CSV 格式**
  - 使用 `mgf2csv.dbond_m.py` 或 `mgf2csv.dbond_s.py` 将标注后的 MGF 文件转换为 CSV 文件
4. **训练和评估**
  - 使用生成的 CSV 数据进行下游模型的训练和评估

如需自定义参数或批量处理，可参考各脚本内的函数调用方式
