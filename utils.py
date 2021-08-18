import json
import os
import pickle
import random
import re

import numpy as np
import torch
from tqdm import tqdm


def brier_multi(targets, probs):
    # https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    targets, probs = np.array(targets), np.array(probs)
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


def make_tuple(exp):
    assert "(" in exp and ")" in exp and exp.count(",") == 1
    exp = [el.strip() for el in exp.strip()[1:-1].split(",")]
    return exp


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def recall_x_at_k(score_list, x, k, answer_index):
    assert len(score_list) == x
    sorted_score_index = np.array(score_list).argsort()[::-1]
    assert answer_index in sorted_score_index
    return int(answer_index in sorted_score_index[:k])


class SelectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        setname: str,
        max_seq_len: int = 300,
        num_candidate: int = 10,
        uttr_token: str = "[UTTR]",
        txt_save_fname: str = None,
        tensor_save_fname: str = None,
        corrupted_context_dataset=None,
        # add_nota_in_every_candidate=False,
    ):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.uttr_token = uttr_token
        assert setname in ["train", "dev", "test"]
        txt_save_fname, tensor_save_fname = (
            txt_save_fname.format(setname),
            tensor_save_fname.format(setname),
        )
        # self.add_nota = add_nota_in_every_candidate
        selection_dataset = self._get_selection_dataset(
            raw_dataset, num_candidate, txt_save_fname, corrupted_context_dataset
        )
        # if self.add_nota:
        #    for el in selection_dataset:
        #        assert "[NOTA]" in el
        self.feature = self._tensorize_selection_dataset(
            selection_dataset, tensor_save_fname, num_candidate
        )

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _tensorize_selection_dataset(
        self, selection_dataset, tensor_save_fname, num_candidate
    ):
        if os.path.exists(tensor_save_fname):
            print(f"{tensor_save_fname} exist!")
            with open(tensor_save_fname, "rb") as f:
                return pickle.load(f)
        print("make {}".format(tensor_save_fname))
        ids_list = [[] for _ in range(num_candidate)]
        masks_list = [[] for _ in range(num_candidate)]
        labels = []
        print("Tensorize...")
        for sample_idx, sample in enumerate(tqdm(selection_dataset)):
            assert len(sample) == 1 + num_candidate and all(
                [isinstance(el, str) for el in sample]
            )
            context, candidates = sample[0], sample[1:]
            assert len(candidates) == num_candidate

            encoded = self.tokenizer(
                [context] * num_candidate,
                text_pair=candidates,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            encoded_ids, encoded_mask = encoded["input_ids"], encoded["attention_mask"]
            assert len(encoded_ids) == len(encoded_mask) == num_candidate
            for candi_idx in range(num_candidate):
                ids_list[candi_idx].append(encoded_ids[candi_idx])
                masks_list[candi_idx].append(encoded_mask[candi_idx])
            labels.append(0)

        assert len(list(set([len(el) for el in ids_list]))) == 1
        assert len(list(set([len(el) for el in masks_list]))) == 1
        ids_list = [torch.stack(el) for el in ids_list]
        masks_list = [torch.stack(el) for el in masks_list]
        labels = torch.tensor(labels)
        data = ids_list + masks_list + [labels]
        assert len(data) == 1 + 2 * num_candidate
        with open(tensor_save_fname, "wb") as f:
            pickle.dump(data, f)
        return data

    def _get_selection_dataset(
        self, raw_dataset, num_candidate, txt_save_fname, corrupted_context_dataset
    ):
        print("Selection filename: {}".format(txt_save_fname))
        if os.path.exists(txt_save_fname):
            print(f"{txt_save_fname} exist!")
            with open(txt_save_fname, "rb") as f:
                return pickle.load(f)

        selection_dataset = self._make_selection_dataset(
            raw_dataset, num_candidate, corrupted_context_dataset
        )
        os.makedirs(os.path.dirname(txt_save_fname), exist_ok=True)
        with open(txt_save_fname, "wb") as f:
            pickle.dump(selection_dataset, f)
        return selection_dataset

    def _make_selection_dataset(
        self, raw_dataset, num_candidate, corrupted_context_dataset
    ):
        """
        Returns:
            datset: List of [context(str), positive_response(str), negative_response_1(str), (...) negative_response_(num_candidate-1)(str)]
        """
        assert isinstance(raw_dataset, list) and all(
            [isinstance(el, list) for el in raw_dataset]
        )
        print(f"Serialized selection not exist. Make new file...")
        dataset = []
        all_responses = []
        for idx, conv in enumerate(tqdm(raw_dataset)):
            slided_conversation = self._slide_conversation(conv)
            # Check the max sequence length
            for single_conv in slided_conversation:
                assert len(single_conv) == 2 and all(
                    [isinstance(el, str) for el in single_conv]
                )
                concat_single_conv = " ".join(single_conv)
                if len(self.tokenizer.tokenize(concat_single_conv)) + 3 <= 300:
                    dataset.append(single_conv)
            all_responses.extend([el[1] for el in slided_conversation])

        if corrupted_context_dataset is not None:
            print("Samples with corrupted context are also included in training")
            print("Before: {}".format(len(dataset)))
            half_sampled_corrupt_sample = random.sample(
                corrupted_context_dataset, int(len(dataset) / 2)
            )
            for corrupted_sample in tqdm(half_sampled_corrupt_sample):
                changed_context = self.tokenizer.decode(
                    corrupted_sample["changed_context"]
                ).strip()
                assert isinstance(changed_context, str)
                assert "[CLS]" == changed_context[:5]
                assert "[SEP]" == changed_context[-5:]
                tmp_text = changed_context[5:-5].strip()
                assert len(self.tokenizer.tokenize(tmp_text)) + 2 <= 300
                dataset.append([tmp_text, "[NOTA]"])
            print("After: {}".format(len(dataset)))

        for idx, el in enumerate(dataset):
            sampled_random_negative = random.sample(all_responses, num_candidate)
            if el[1] in sampled_random_negative:
                sampled_random_negative.remove(el[1])
            sampled_random_negative = sampled_random_negative[: num_candidate - 1]
            dataset[idx].extend(sampled_random_negative)

            # if not self.add_nota:
            #     sampled_random_negative = sampled_random_negative[: num_candidate - 1]
            #     dataset[idx].extend(sampled_random_negative)
            # else:
            #     sampled_random_negative = ["[NOTA]"] + sampled_random_negative[: num_candidate - 2]
            #     dataset[idx].extend(sampled_random_negative)
            assert len(dataset[idx]) == 1 + num_candidate
            assert all([isinstance(txt, str) for txt in dataset[idx]])
        return dataset

    def _slide_conversation(self, conversation):
        """
        multi-turn utterance로 이루어진 single conversation을 여러 개의 "context-response" pair로 만들어 반환
        """
        assert isinstance(conversation, list) and all(
            [isinstance(el, str) for el in conversation]
        )
        pairs = []
        for idx in range(len(conversation) - 1):
            context, response = conversation[: idx + 1], conversation[idx + 1]
            pairs.append([self.uttr_token.join(context), response])
        return pairs


def get_uttr_token():
    return "[UTTR]"


def get_nota_token():
    return "[NOTA]"


def dump_config(args):
    with open(os.path.join(args.exp_path, "config.json"), "w") as f:
        json.dump(vars(args), f)


def write2tensorboard(writer, value, setname, step):
    for k, v in value.items():
        writer.add_scalars(k, {setname: v}, step)
    writer.flush()


def save_model(model, epoch, model_path):
    try:
        torch.save(
            model.module.state_dict(),
            os.path.join(model_path, f"epoch-{epoch}.pth"),
        )
    except:
        torch.save(
            model.state_dict(),
            os.path.join(model_path, f"epoch-{epoch}.pth"),
        )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_model(model, model_path, epoch, len_tokenizer):
    model.bert.resize_token_embeddings(len_tokenizer)
    model.load_state_dict(torch.load(model_path + f"/epoch-{epoch}.pth"))
    return model


def make_random_negative_for_multi_ref(multiref_original, num_neg=30):
    for idx, item in enumerate(multiref_original):
        context, responses = item
        sample = random.sample(range(len(multiref_original)), num_neg + 1)
        if idx in sample:
            sample.remove(idx)
        else:
            sample = sample[:-1]
        responses = [multiref_original[sample_idx][1] for sample_idx in sample]
        responses = [el for el1 in responses for el in el1]
        assert all([isinstance(el, str) for el in responses])
        negative = random.sample(responses, num_neg)
        multiref_original[idx].append(negative)
    return multiref_original
