import argparse
import json
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (BertConfig, BertForNextSentencePrediction, BertModel,
                          BertTokenizer)

from preprocess_dataset import (get_dd_corpus, get_dd_multiref_testset)
from selection_model import BertSelect, BertSelectAuxilary
from utils import (SelectionDataset, dump_config,
                   get_nota_token, get_uttr_token, load_model, recall_x_at_k,
                   set_random_seed, str2bool)


def main(args):
    set_random_seed(42)

    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    UTTR_TOKEN = get_uttr_token()
    NOTA_TOKEN = get_nota_token()

    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN, NOTA_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model_list = []
    seed_list = [42] if args.model != "ensemble" else [42, 43, 44, 45, 46]
    for seed in seed_list:
        bert = BertModel.from_pretrained("bert-base-uncased")
        bert.resize_token_embeddings(len(tokenizer))
        if args.is_aux_model:
            model = BertSelectAuxilary(bert)
        else:
            model = BertSelect(bert)
        model = load_model(model, args.model_path.format(seed), 0, len(tokenizer))

    model.to(device)
    model_list.append(model)

    print("usual testset")
    
    txt_fname = (
        "./data/selection/text_cand{}".format(args.retrieval_candidate_num)
        + "_{}.pck"
    )
    tensor_fname = (
        "./data/selection/tensor_cand{}".format(args.retrieval_candidate_num)
        + "_{}.pck"
    )
    raw_dataset = get_dd_corpus(
        "validation" if args.setname == "valid" else args.setname
    )

    selection_dataset = SelectionDataset(
        raw_dataset,
        tokenizer,
        args.setname,
        300,
        args.retrieval_candidate_num,
        UTTR_TOKEN,
        txt_fname,
        tensor_fname,
    )

    total_item_list = []
    dataset_length = len(selection_dataset)

    for idx in tqdm(range(dataset_length)):
        pred_list_for_current_context = []
        uncertainty_list_for_current_context = []

        sample = [el[idx] for el in selection_dataset.feature]

        assert len(sample) == 2 * args.retrieval_candidate_num + 1

        ids = torch.stack([sample[i] for i in range(args.retrieval_candidate_num)]).to(device)
        mask = torch.stack(
            [
                sample[i + args.retrieval_candidate_num]
                for i in range(args.retrieval_candidate_num)
            ]
        ).to(device)
        prediction_list = []
        with torch.no_grad():
            if args.is_aux_model:
                assert len(model_list) == 1
                with torch.no_grad():
                    model = model_list[0]
                    prediction_list.append(model.predict(ids, mask).cpu().numpy())
                prediction_list = np.array(prediction_list)
                pred_list_for_current_context = np.mean(prediction_list, 0)
                uncertainty_list_for_current_context = np.var(prediction_list, 0)
            else:
                if args.model == "mcdrop":
                    assert len(model_list) == 1
                    model = model_list[0]
                    model.train()
                    for forward_pass in range(5):
                        with torch.no_grad():
                            prediction_list.append(
                                [float(el) for el in model(ids, mask).cpu().numpy()]
                            )
                    prediction_list = np.array(prediction_list)
                    pred_list_for_current_context = np.mean(prediction_list, 0)
                    uncertainty_list_for_current_context = np.var(prediction_list, 0)
                else:
                    assert args.model in ["ensemble", "select", "nopt"]
                    for model in model_list:
                        with torch.no_grad():
                            prediction_list.append(
                                [float(el) for el in model(ids, mask).cpu().numpy()]
                            )
                    prediction_list = np.array(prediction_list)
                    pred_list_for_current_context = np.mean(prediction_list, 0)
                    uncertainty_list_for_current_context = np.var(prediction_list, 0)

        pred_list_for_current_context = [
            float(el) for el in pred_list_for_current_context
        ]
        uncertainty_list_for_current_context = [
            float(el) for el in uncertainty_list_for_current_context
        ]
        assert (
            len(pred_list_for_current_context)
            == len(uncertainty_list_for_current_context)
            == args.retrieval_candidate_num
        )

        total_item_list.append(
            {
                "pred": pred_list_for_current_context,
                "uncertainty": uncertainty_list_for_current_context,
            }
        )

    with open(args.output_fname, "w") as f:
        for l in total_item_list:
            json.dump(l, f)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--corpus", default="dd", choices=["persona", "dd"])
    parser.add_argument("--setname", default="test", choices=["valid", "test"])
    parser.add_argument("--log_path", type=str, default="result")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./logs/select_batch12_candi2_seed{}/model",
    )
    parser.add_argument(
        "--retrieval_candidate_num",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--model",
        default="select",
        help="compared method",
        choices=["select", "mcdrop", "ensemble", "nopt", "uw"],
    )
    parser.add_argument(
        "--direct_threshold",
        type=float,
        default=-1,
        help="baseline threshold",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed during training",
    )
    parser.add_argument("--is_aux_model", type=str2bool, default=False)

    args = parser.parse_args()

    assert len(args.model_path.split("/")) == 4

    args.exp_name = f"{args.model}-candi{args.retrieval_candidate_num}-{args.setname}"

    args.log_path = os.path.join(args.log_path, args.corpus)

    os.makedirs(args.log_path, exist_ok=True)
    args.output_fname = os.path.join(args.log_path, args.exp_name) + ".json"
    print("\n", args.output_fname, "\n")

    assert not os.path.exists(args.output_fname)
    os.makedirs(os.path.dirname(args.output_fname), exist_ok=True)
    
    if args.model == "nopt":
        assert "randinit" in args.model_path

    main(args)
