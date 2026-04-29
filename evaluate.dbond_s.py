import torch
from dbond_s import Model as Net
from dbond_s import focal_loss
from data_utils_dbond_s import PepDataset, collate_callback
from sklearn.metrics import (
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import tqdm
from typing import List, Dict, Callable, Tuple
import numpy
import argparse
import random
import os
import pandas
import yaml
import logging
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# add args
parser = argparse.ArgumentParser(description="Args for dbond-s evaluate")
parser.add_argument(
    "--in_model_weight_path",
    type=str,
    help="Path to the model weight file",
    default="./best_model/dbond_s/2025_11_29_18_46_default_5.pt",
)
parser.add_argument(
    "--in_model_comfig_path",
    type=str,
    help="Path to the model config file",
    default="./dbond_s_config/default.yaml",
)
parser.add_argument(
    "--in_csv_to_predict_path",
    type=str,
    help="Path to the input csv file to predict",
    default="./dataset/dbond_s.test.csv",
)
parser.add_argument(
    "--in_csv_for_multi_label_path",
    type=str,
    help="Path to the input csv file for multi label metric compute",
    default="./dataset/dataset.fbr.csv",
)
parser.add_argument(
    "--out_single_label_pred_dir",
    type=str,
    help="Path to save the single label prediction result",
    default="./result/pred/dbond_s/single/",
)
parser.add_argument(
    "--out_multi_label_pred_dir",
    type=str,
    help="Path to save the multi label prediction result",
    default="./result/pred/dbond_s/multi/",
)
parser.add_argument(
    "--out_single_label_metric_dir",
    type=str,
    help="Path to save the single label metric result",
    default="./result/metric/dbond_s/single/",
)
parser.add_argument(
    "--out_multi_label_metric_dir",
    type=str,
    help="Path to save the multi label metric result",
    default="./result/metric/dbond_s/multi/",
)

args = parser.parse_args()
model_weight_path = args.in_model_weight_path
model_config_path = args.in_model_comfig_path
csv_to_predict_path = args.in_csv_to_predict_path
csv_for_multi_label_metric_path = args.in_csv_for_multi_label_path

if not model_weight_path or not model_config_path:
    logging.error("Please provide the model weight path and model config path")
    exit(1)
if not os.path.exists(model_weight_path) or not os.path.exists(model_config_path):
    logging.error(
        f"Model weight path: {model_weight_path} or model config path:{model_config_path} does not exist"
    )
    exit(1)
if not csv_to_predict_path or not csv_for_multi_label_metric_path:
    logging.error(
        "Please provide the csv file to predict and the csv file for multi label metric compute"
    )
    exit(1)
if not os.path.exists(csv_to_predict_path) or not os.path.exists(
    csv_for_multi_label_metric_path
):
    logging.error(
        f"CSV file to predict:{csv_to_predict_path} or CSV file for multi label metric compute:{csv_for_multi_label_metric_path} does not exist"
    )
    exit(1)

model_config = yaml.safe_load(open(model_config_path, "r"))
model_config["csv"]["test_dataset_path"] = csv_to_predict_path

TAG = model_config["tag"]
out_single_label_pred_path = args.out_single_label_pred_dir

out_single_label_pred_path = os.path.join(out_single_label_pred_path, f"{TAG}.pred.csv")
out_multi_label_pred_path = args.out_multi_label_pred_dir

out_multi_label_pred_path = os.path.join(out_multi_label_pred_path, f"{TAG}.pred.csv")
out_single_label_metric_path = args.out_single_label_metric_dir

out_single_label_metric_path = os.path.join(
    out_single_label_metric_path, f"{TAG}_metric.csv"
)
out_multi_label_metric_path = args.out_multi_label_metric_dir

out_multi_label_metric_path = os.path.join(
    out_multi_label_metric_path, f"{TAG}_metric.csv"
)

logging.info(f"Model weight path: {model_weight_path}")
logging.info(f"Model config path: {model_config_path}")
logging.info(f"CSV to predict path: {csv_to_predict_path}")
logging.info(
    f"CSV path for multi label metric compute: {csv_for_multi_label_metric_path}"
)
logging.info(f"Out single label pred dir: {out_single_label_pred_path}")
logging.info(f"Out multi label pred dir: {out_multi_label_pred_path}")
logging.info(f"Out single label metric path: {out_single_label_metric_path}")
logging.info(f"Out multi label metric path: {out_multi_label_metric_path}")
###############################################################################
pad_len = int(model_config["seq"]["max_len"]) - 1

condition_col = ["rt", "intensity", "scan_num", "seq", "nce", "charge"]
tb_col = "tb"


best_model_check_point: Dict[str, Dict]
best_model_check_point = torch.load(model_weight_path)

logging.info("=" * 10 + "train args" + "=" * 10)
for k, v in best_model_check_point["train_args"].items():
    logging.info(f"{k:15}\t{v}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("=" * 10 + str(device) + "=" * 10)


torch.manual_seed(model_config["train_args"]["seed"])
torch.cuda.manual_seed_all(model_config["train_args"]["seed"])
numpy.random.seed(model_config["train_args"]["seed"])
random.seed(model_config["train_args"]["seed"])

dataset = PepDataset(model_config, split='test')


dataloader = torch.utils.data.DataLoader(
    dataset,
    pin_memory=True,
    shuffle=False,
    batch_size=model_config["train_args"]["batch_size"],
    collate_fn=collate_callback,
    num_workers=model_config["train_args"]["dataloader_workers"],
)

model = Net(model_config)

model.load_state_dict(best_model_check_point["model_state_dict"])
logging.info(str(model))
model.to(device)

loss_func: Callable

if model_config["train_args"]["loss_type"].lower() == "ce":
    loss_func = lambda logits, labels: torch.nn.functional.cross_entropy(
        logits, labels, reduction="mean", **model_config["train_args"]["loss_args"]
    )
elif model_config["train_args"]["loss_type"].lower() == "focal":
    loss_func = lambda logits, labels: focal_loss(
        logits, labels, reduction="mean", **model_config["train_args"]["loss_args"]
    )


def evaluate(
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[dict, List[int], List[int]]:

    model.eval()

    loss_sum = []
    preds = []
    preds_probs = []
    trues = []

    seq_index_batch: torch.Tensor
    seq_padding_mask_batch: torch.Tensor
    bond_index_batch: torch.Tensor
    bond_vec_batch: torch.Tensor
    state_vec_batch: torch.Tensor
    env_vec_batch: torch.Tensor
    label_real_batch: torch.Tensor

    with tqdm.tqdm(dataloader, total=len(dataloader), unit="batch") as loop:
        for (
            seq_index_batch,
            seq_padding_mask_batch,
            bond_index_batch,
            bond_vec_batch,
            state_vec_batch,
            env_vec_batch,
            label_real_batch,
        ) in loop:
            loop.set_description(f"EVAL")

            seq_index_batch = seq_index_batch.to(device)
            seq_padding_mask_batch = seq_padding_mask_batch.to(device)
            bond_index_batch = bond_index_batch.to(device)
            bond_vec_batch = bond_vec_batch.to(device)
            state_vec_batch = state_vec_batch.to(device)
            env_vec_batch = env_vec_batch.to(device)
            label_real_batch = label_real_batch.to(device)

            label_predict_batch = model.forward(
                seq_index_batch=seq_index_batch,
                seq_padding_mask_batch=seq_padding_mask_batch,
                bond_index_batch=bond_index_batch,
                bond_vec_batch=bond_vec_batch,
                state_vec_batch=state_vec_batch,
                env_vec_batch=env_vec_batch,
            )
            loss: torch.Tensor = loss_func(label_predict_batch, label_real_batch)

            loss_sum.append(label_real_batch.shape[0] * loss.item())

            # logging.info(label_predict_batch.shape)
            label_prob_batch = torch.nn.functional.softmax(label_predict_batch, dim=1)

            label_predict_batch = label_predict_batch.argmax(dim=1)

            preds.extend(label_predict_batch.detach().cpu().numpy())
            preds_probs.extend(label_prob_batch[:, 1].detach().cpu().numpy())
            trues.extend(label_real_batch.detach().cpu().numpy())
            loop.set_postfix({"loss": loss.item()})
            # break

    mean_loss = numpy.sum(loss_sum) / len(dataset)

    sklearn_accuracy = accuracy_score(trues, preds)
    sklearn_auc = roc_auc_score(trues, preds_probs)
    sklearn_ap = average_precision_score(trues, preds_probs)

    sklearn_precision_label_0, sklearn_precision_label_1 = precision_score(
        trues, preds, average=None
    )
    sklearn_recall_label_0, sklearn_recall_label_1 = recall_score(
        trues, preds, average=None
    )
    sklearn_f1_label_0, sklearn_f1_label_1 = f1_score(trues, preds, average=None)

    metrics_dict = {
        "Loss": mean_loss,
        "accuracy": sklearn_accuracy,
        "AUC": sklearn_auc,
        "AP": sklearn_ap,
        "precision": (sklearn_precision_label_0 + sklearn_precision_label_1) / 2,
        "recall": (sklearn_recall_label_0 + sklearn_recall_label_1) / 2,
        "f1": (sklearn_f1_label_0 + sklearn_f1_label_1) / 2,
        "Label 0: precision": sklearn_precision_label_0,
        "Label 1: precision": sklearn_precision_label_1,
        "Label 0: recall": sklearn_recall_label_0,
        "Label 1: recall": sklearn_recall_label_1,
        "Label 0: f1": sklearn_f1_label_0,
        "Label 1: f1": sklearn_f1_label_1,
    }

    return metrics_dict, trues, preds


metric, trues, preds = evaluate(dataloader=dataloader)
###############################################################################
logging.info("=" * 10 + "single label metric" + "=" * 10)
for k, v in metric.items():
    logging.info(f"{k:30}\t{v:.4}")
metric_df = pandas.DataFrame(list(metric.items()), columns=["metric", "value"])
logging.info(f"save single metric result to {out_single_label_metric_path}")
metric_df.to_csv(out_single_label_metric_path, index=False)

logging.info(f"process single label pred result")
bond_csv_df = pandas.read_csv(csv_to_predict_path, na_filter=None)
bond_csv_df["true"] = trues
bond_csv_df["pred"] = preds
logging.info(f"single bond predict finish")
logging.info(f"save single label pred result to {out_single_label_pred_path}")
bond_csv_df.to_csv(out_single_label_pred_path, index=False)

logging.info(f"process multi label pred result")
fbr_csv_df: pandas.DataFrame = pandas.read_csv(
    csv_for_multi_label_metric_path, na_filter=None
)
pep_list = dataset.df["seq"].unique()
fbr_csv_df = fbr_csv_df[fbr_csv_df["seq"].isin(pep_list)]


def process_row(row: pandas.Series):
    sub_df: pandas.DataFrame = bond_csv_df
    for k in condition_col:
        condition = sub_df[k].values == row[k]
        sub_df = sub_df[condition]
    assert (
        len(sub_df) == row[tb_col]
    ), f"condition is not unique !\n{condition_col}\ntb = {row[tb_col]}\nlen(sub_df) = {len(sub_df)}"
    sub_df.sort_values(by="bond_pos")
    label_true_list = sub_df["true"].tolist()
    label_pred_list = sub_df["pred"].tolist()
    multi_label_true_str = ";".join(map(str, label_true_list))
    multi_label_pred_str = ";".join(map(str, label_pred_list))
    return multi_label_true_str, multi_label_pred_str


fbr_csv_df["true_multi"], fbr_csv_df["pred_multi"] = zip(
    *fbr_csv_df.parallel_apply(lambda row: process_row(row), axis=1)
)

logging.info(f"multi bond result precess finish")
logging.info(f"save multi label pred result to {out_multi_label_pred_path}")
fbr_csv_df.to_csv(out_multi_label_pred_path, index=False)

logging.info(f"compute multi label metric")
multi_label_true_str_list = fbr_csv_df["true_multi"].tolist()
multi_label_pred_str_list = fbr_csv_df["pred_multi"].tolist()


def masked_metric(
    multi_label_true_str_list: List[str], multi_label_pred_str_list: List[str]
):

    import numpy as np

    multi_label_true_list_list = [
        list(map(int, item.split(";"))) for item in multi_label_true_str_list
    ]
    multi_label_true_list_list = [
        item + [0] * (pad_len - len(item)) for item in multi_label_true_list_list
    ]
    multi_label_true_list_list = np.array(multi_label_true_list_list)
    multi_label_pred_list_list = [
        list(map(int, item.split(";"))) for item in multi_label_pred_str_list
    ]
    multi_label_pred_list_list = [
        item + [0] * (pad_len - len(item)) for item in multi_label_pred_list_list
    ]
    multi_label_pred_list_list = np.array(multi_label_pred_list_list)
    n = len(multi_label_true_list_list)
    m = len(multi_label_true_list_list[0])
    logging.info(f"n: {n}\tm:{m}")

    gt = multi_label_true_list_list
    predict = multi_label_pred_list_list
    import multi_label_metrics

    subset_acc = multi_label_metrics.example_subset_accuracy(gt, predict)
    ex_acc = multi_label_metrics.example_accuracy(gt, predict)
    ex_precision = multi_label_metrics.example_precision(gt, predict)
    ex_recall = multi_label_metrics.example_recall(gt, predict)
    ex_f1 = multi_label_metrics.example_f1(gt, predict)

    lab_acc_ma = multi_label_metrics.label_accuracy_macro(gt, predict)
    lab_acc_mi = multi_label_metrics.label_accuracy_micro(gt, predict)
    lab_precision_ma = multi_label_metrics.label_precision_macro(gt, predict)
    lab_precision_mi = multi_label_metrics.label_precision_micro(gt, predict)
    lab_recall_ma = multi_label_metrics.label_recall_macro(gt, predict)
    lab_recall_mi = multi_label_metrics.label_recall_micro(gt, predict)
    lab_f1_ma = multi_label_metrics.label_f1_macro(gt, predict)
    lab_f1_mi = multi_label_metrics.label_f1_micro(gt, predict)

    metric["subset_acc"] = subset_acc
    metric["ex_acc"] = ex_acc
    metric["ex_precision"] = ex_precision
    metric["ex_recall"] = ex_recall
    metric["ex_f1"] = ex_f1
    metric["lab_acc_ma"] = lab_acc_ma
    metric["lab_acc_mi"] = lab_acc_mi
    metric["lab_precision_ma"] = lab_precision_ma
    metric["lab_precision_mi"] = lab_precision_mi
    metric["lab_recall_ma"] = lab_recall_ma
    metric["lab_recall_mi"] = lab_recall_mi
    metric["lab_f1_ma"] = lab_f1_ma
    metric["lab_f1_mi"] = lab_f1_mi

    logging.info("subset acc\t\t\t%.4f" % subset_acc)
    logging.info("example acc\t\t\t%.4f" % ex_acc)
    logging.info("example precision\t\t%.4f" % ex_precision)
    logging.info("example recall\t\t%.4f" % ex_recall)
    logging.info("example f1\t\t\t%.4f" % ex_f1)
    logging.info("label acc macro\t\t%.4f" % lab_acc_ma)
    logging.info("label acc micro\t\t%.4f" % lab_acc_mi)
    logging.info("label prec macro\t\t%.4f" % lab_precision_ma)
    logging.info("label prec micro\t\t%.4f" % lab_precision_mi)
    logging.info("label rec macro\t\t%.4f" % lab_recall_ma)
    logging.info("label rec micro\t\t%.4f" % lab_recall_mi)
    logging.info("label f1 macro\t\t%.4f" % lab_f1_ma)
    logging.info("label f1 micro\t\t%.4f" % lab_f1_mi)


masked_metric(multi_label_true_str_list, multi_label_pred_str_list)
logging.info(f"save multi label metric result to {out_multi_label_metric_path}")
metric_df = pandas.DataFrame(list(metric.items()), columns=["metric", "value"])
metric_df.to_csv(out_multi_label_metric_path, index=False)
logging.info("evaluate finished")
