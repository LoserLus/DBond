import torch
from dbond_m import Model as Net
from dbond_m import multilabel_categorical_crossentropy
from data_utils_dbond_m import PepDataset, collate_callback
import tqdm
from typing import List, Dict, Callable, Tuple
import numpy
import argparse
import random
import os
import pandas
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
parser = argparse.ArgumentParser(description="Args for dbond-m evaluate")
parser.add_argument(
    "--in_model_weight_path",
    type=str,
    help="Path to the model weight file",
    default="./best_model/dbond_m/2025_11_25_19_25_default_0.pt",
)
parser.add_argument(
    "--in_model_comfig_path",
    type=str,
    help="Path to the model config file",
    default="./dbond_m_config/default.yaml",
)
parser.add_argument(
    "--in_csv_to_predict_path",
    type=str,
    help="Path to the input csv file to predict",
    default="./dataset/dbond_m.test.csv",
)
parser.add_argument(
    "--out_multi_label_pred_dir",
    type=str,
    help="Path to save the multi label prediction result",
    default="./result/pred/dbond_m/multi/",
)

parser.add_argument(
    "--out_multi_label_metric_dir",
    type=str,
    help="Path to save the multi label metric result",
    default="./result/metric/dbond_m/multi/",
)
args = parser.parse_args()
model_weight_path = args.in_model_weight_path
model_config_path = args.in_model_comfig_path
csv_to_predict_path = args.in_csv_to_predict_path

if not model_weight_path or not model_config_path:
    logging.error("Please provide the model weight path and model config path")
    exit(1)
if not os.path.exists(model_weight_path) or not os.path.exists(model_config_path):
    logging.error(
        f"Model weight path: {model_weight_path} or model config path:{model_config_path} does not exist"
    )
    exit(1)
if not csv_to_predict_path:
    logging.error("Please provide the csv file to predict")
    exit(1)
if not os.path.exists(csv_to_predict_path):
    logging.error(f"CSV file to predict:{csv_to_predict_path} does not exist")
    exit(1)

model_config = yaml.safe_load(open(model_config_path, "r"))
model_config["csv"]["test_dataset_path"] = csv_to_predict_path
TAG = model_config["tag"]
out_multi_label_pred_path = args.out_multi_label_pred_dir

out_multi_label_pred_path = os.path.join(out_multi_label_pred_path, f"{TAG}.pred.csv")

out_multi_label_metric_path = args.out_multi_label_metric_dir

out_multi_label_metric_path = os.path.join(
    out_multi_label_metric_path, f"{TAG}_metric.csv"
)
logging.info(f"Model weight path: {model_weight_path}")
logging.info(f"Model config path: {model_config_path}")
logging.info(f"CSV to predict path: {csv_to_predict_path}")
logging.info(f"Out multi label pred dir: {out_multi_label_pred_path}")
logging.info(f"Out multi label metric path: {out_multi_label_metric_path}")

################################################################################
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
    loss_func = lambda logits, labels: torch.nn.functional.multilabel_soft_margin_loss(
        logits, labels
    )
elif model_config["train_args"]["loss_type"].lower() == "zlpr":
    loss_func = lambda logits, labels: multilabel_categorical_crossentropy(
        labels, logits
    )


def evaluate(
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[dict, List[List[int]], List[List[int]]]:

    model.eval()
    loss_sum = []
    predict = []
    predict_probs = []
    gt = []
    seq_index_batch: torch.Tensor
    seq_padding_mask_batch: torch.Tensor

    state_vec_batch: torch.Tensor
    env_vec_batch: torch.Tensor
    label_real_batch: torch.Tensor

    with tqdm.tqdm(dataloader, total=len(dataloader), unit="batch") as loop:
        for (
            seq_index_batch,
            seq_padding_mask_batch,
            state_vec_batch,
            env_vec_batch,
            label_real_batch,
        ) in loop:
            loop.set_description(f"EVAL")

            seq_index_batch = seq_index_batch.to(device)
            seq_padding_mask_batch = seq_padding_mask_batch.to(device)
            state_vec_batch = state_vec_batch.to(device)
            env_vec_batch = env_vec_batch.to(device)
            label_real_batch = label_real_batch.to(device)

            logits_predict_batch = model.forward(
                seq_index_batch=seq_index_batch,
                seq_padding_mask_batch=seq_padding_mask_batch,
                state_vec_batch=state_vec_batch,
                env_vec_batch=env_vec_batch,
            )

            loss: torch.Tensor = loss_func(logits_predict_batch, label_real_batch)

            loss_sum.append(label_real_batch.shape[0] * loss.item())

            label_prob_batch = torch.nn.functional.sigmoid(logits_predict_batch)

            label_predict_batch = (label_prob_batch > 0.5).long()

            predict.extend(label_predict_batch.detach().cpu().numpy())
            predict_probs.extend(label_prob_batch.detach().cpu().numpy())
            gt.extend(label_real_batch.detach().cpu().numpy())
            loop.set_postfix({"loss": loss.item()})
            # break

    mean_loss: float

    mean_loss = numpy.sum(loss_sum) / len(dataset)

    import multi_label_metrics

    gt = numpy.vstack(gt)
    predict = numpy.vstack(predict)
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

    metrics_dict = {
        "Loss": mean_loss,
        "subset_acc": subset_acc,
        "ex_acc": ex_acc,
        "ex_precision": ex_precision,
        "ex_recall": ex_recall,
        "ex_f1": ex_f1,
        "lab_acc_ma": lab_acc_ma,
        "lab_acc_mi": lab_acc_mi,
        "lab_precision_ma": lab_precision_ma,
        "lab_precision_mi": lab_precision_mi,
        "lab_recall_ma": lab_recall_ma,
        "lab_recall_mi": lab_recall_mi,
        "lab_f1_ma": lab_f1_ma,
        "lab_f1_mi": lab_f1_mi,
    }
    return metrics_dict, gt, predict


metric, trues, preds = evaluate(dataloader=dataloader)
################################################################################
logging.info("=" * 10 + "multi label metric" + "=" * 10)
for k, v in metric.items():
    logging.info(f"{k:30}\t{v:.4}")
metric_df = pandas.DataFrame(list(metric.items()), columns=["metric", "value"])
logging.info(f"save multi metric result to {out_multi_label_metric_path}")
metric_df.to_csv(out_multi_label_metric_path, index=False)

logging.info(f"process multi label pred result")
bond_csv_df = pandas.read_csv(csv_to_predict_path, na_filter=None)
bond_csv_df["true"] = list(map(lambda x: ";".join(list(map(str, x))), trues))
bond_csv_df["pred"] = list(map(lambda x: ";".join(list(map(str, x))), preds))

bond_csv_df.to_csv(out_multi_label_pred_path, index=False)
logging.info(f"save single label pred result to {out_multi_label_pred_path}")
logging.info("evaluate finished")
