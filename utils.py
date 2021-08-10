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


def corrupt_context_wordlevel_for_auxilary(
    ids,
    mask,
    use_attn: bool,
    corrupt_ratio: float,
    sep_id,
    skip_token_ids,
    device=None,
    model=None,
):
    numpy_ids = np.array(ids)
    numpy_mask = np.array(mask)

    bs, max_len = numpy_ids.shape
    context_end_indices = np.where(numpy_ids == sep_id)[1].reshape(bs, 2)[:, 0]

    if use_attn:
        with torch.no_grad():
            _, attentions = model.get_attention(ids.to(device), mask.to(device))
            # 12, [bs,12,300,300]
            attention_output = [el.cpu().numpy() for el in attentions]

            for seq_idx, seq in enumerate(numpy_ids):
                this_seq_attention_output = sum(
                    [sum(sum(el[seq_idx])) for el in attention_output]
                )
                attn_score = [
                    tmp_score
                    if seq[tmp_idx] not in skip_token_ids
                    and tmp_idx > context_end_indices[seq_idx]
                    else 0.0
                    for tmp_idx, tmp_score in enumerate(this_seq_attention_output)
                ]

                sorted_score_indices = np.argsort(attn_score)[::-1]
                selected_indices = sorted(
                    sorted_score_indices[
                        : int((context_end_indices[seq_idx] - 1) * corrupt_ratio)
                    ]
                )
                modified_ids = numpy_ids[seq_idx].copy().tolist()
                modified_mask = numpy_mask[seq_idx].copy().tolist()
                for deleted_order, deleted_index in enumerate(selected_indices):
                    modified_ids.pop(deleted_index - deleted_order)
                    modified_ids.append(0)
                    modified_mask.pop(0)
                    modified_mask.append(0)
                assert (
                    len(modified_ids) == len(numpy_ids[seq_idx]) == len(modified_mask)
                )
                numpy_ids[seq_idx], numpy_mask[seq_idx] = modified_ids, modified_mask
            return torch.tensor(numpy_ids), torch.tensor(numpy_mask)
    else:
        for seq_idx, seq in enumerate(numpy_ids):
            selected_indices = [i + 1 for i in range(context_end_indices[seq_idx] - 1)]
            selected_indices = sorted(
                random.sample(
                    selected_indices, int(len(selected_indices) * corrupt_ratio)
                )
            )
            modified_ids = numpy_ids[seq_idx].copy().tolist()
            modified_mask = numpy_mask[seq_idx].copy().tolist()
            for deleted_order, deleted_index in enumerate(selected_indices):
                modified_ids.pop(deleted_index - deleted_order)
                modified_ids.append(0)
                modified_mask.pop(0)
                modified_mask.append(0)
            assert len(modified_ids) == len(numpy_ids[seq_idx]) == len(modified_mask)
            numpy_ids[seq_idx], numpy_mask[seq_idx] = modified_ids, modified_mask
        return torch.tensor(numpy_ids), torch.tensor(numpy_mask)


def make_corrupted_select_dataset(
    uw_data,
    dd_dataset,
    retrieval_candidate_num,
    save_fname,
    tokenizer,
    max_seq_len,
    replace_golden_to_nota,
):
    assert not replace_golden_to_nota
    if os.path.exists(save_fname):
        print("{} exist!".format(save_fname))
        with open(save_fname, "rb") as f:
            return pickle.load(f)
    nota_token = get_nota_token()
    assert isinstance(uw_data, list) and all([len(el) == 2 for el in uw_data])
    responses = [uttr for conv in dd_dataset for uttr in conv[1:]]
    assert all([isinstance(el, str) for el in responses])
    for idx, hist in enumerate(uw_data):
        assert len(hist) == 2 and all([isinstance(el, str) for el in hist])
        assert hist[1] == nota_token or not replace_golden_to_nota
        candidates = random.sample(responses, retrieval_candidate_num - 1)
        uw_data[idx].extend(candidates)

    ids_list = [[] for _ in range(retrieval_candidate_num)]
    masks_list = [[] for _ in range(retrieval_candidate_num)]
    labels = []
    print("Tensorize...")
    for sample_idx, sample in enumerate(tqdm(uw_data)):
        assert len(sample) == 1 + retrieval_candidate_num
        assert all([isinstance(el, str) for el in sample])
        context, candidates = sample[0], sample[1:]
        assert len(candidates) == retrieval_candidate_num
        encoded = tokenizer(
            [context] * retrieval_candidate_num,
            text_pair=candidates,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_ids, encoded_mask = encoded["input_ids"], encoded["attention_mask"]
        assert len(encoded_ids) == len(encoded_mask) == retrieval_candidate_num
        for candi_idx in range(retrieval_candidate_num):
            ids_list[candi_idx].append(encoded_ids[candi_idx])
            masks_list[candi_idx].append(encoded_mask[candi_idx])
        labels.append(0)
    assert len(list(set([len(el) for el in ids_list]))) == 1
    assert len(list(set([len(el) for el in masks_list]))) == 1
    ids_list = [torch.stack(el) for el in ids_list]
    masks_list = [torch.stack(el) for el in masks_list]
    labels = torch.tensor(labels)
    data = ids_list + masks_list + [labels]
    assert len(data) == 1 + 2 * retrieval_candidate_num
    with open(save_fname, "wb") as f:
        pickle.dump(data, f)
    return data


def make_tuple(exp):
    assert "(" in exp and ")" in exp and exp.count(",") == 1
    exp = [el.strip() for el in exp.strip()[1:-1].split(",")]
    return exp


def get_ic_annotation(fname, change_ic_to_original: bool):
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]

    item_list, item = [], {}
    uttr_token = get_uttr_token()

    for line in ls:
        if line == "":
            assert len(item) != 0
            item_list.append(item)
            item = {}
            continue
        if len(item) == 0:
            tmp = [int(el) for el in line.strip().split()]
            assert len(tmp) == 2
            item["remain_context_num"] = tmp[1]
            # item["removed_context_num"] = tmp[1]
            # item["remain_context_num"] = tmp[2]
            continue
        if "uttrs" not in item:
            item["uttrs"] = []
        item["uttrs"].append(line)
    if len(item) != 0:
        item_list.append(item)
    final_output = []
    for item_idx, item in enumerate(item_list):
        # removed_num, remain_num = item["removed_context_num"], item["remain_context_num"]
        remain_num = item["remain_context_num"]
        uttrs = item["uttrs"]
        # assert len(uttrs) in [removed_num + remain_num + 1, removed_num + remain_num]
        assert len(uttrs) == 1 + remain_num
        context = uttrs[:-1]
        response = uttrs[-1]

        # assert len(context) in [remain_num + removed_num, remain_num + removed_num - 1]
        assert len(context) == remain_num
        if not change_ic_to_original:
            context = context[-remain_num:]
            assert len(context) == remain_num
        else:
            raise ValueError
        context = uttr_token.join(context)
        context = context.replace(" ##", "")
        response = response.replace(" ##", "")
        assert "##" not in context
        assert "##" not in response
        final_output.append([context, response])

    return final_output


def get_uw_annotation(fname, change_uw_to_original: bool):
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]
    item_list, item = [], {}
    uttr_token = get_uttr_token()
    original_turn, changed_turn = False, False

    for line_idx, line in enumerate(ls):
        if line == "":
            if changed_turn:
                assert not original_turn
                assert len(item) != 0
                item_list.append(item)
                item = {}
                changed_turn = False
                continue
            elif original_turn:
                assert not changed_turn
                continue
            else:
                print(line_idx)
                print(original_turn, changed_turn)
                print(len(item_list))
                raise ValueError()

        # head
        if len(item) == 0:
            idx, change_num = [int(el) for el in line.split()]
            item["idx"] = idx
            item["num_change"] = change_num
            continue
        # original
        if len(item) == 2:
            original_words = line.split()
            item["original_words"] = original_words
            continue
        # original
        if len(item) == 3:
            changed_words = line.split()
            item["changed_words"] = changed_words
            continue

        if line == "origin":
            assert len(item) == 4
            assert not original_turn and not changed_turn
            original_turn = True
            item["original_uttrs"] = []
            continue
        if line == "changed":
            assert len(item) == 5
            original_turn = False
            assert not original_turn and not changed_turn
            item["changed_uttrs"] = []
            changed_turn = True
            continue
        if original_turn:
            item["original_uttrs"].append(line.strip())
            continue
        if changed_turn:
            item["changed_uttrs"].append(line.strip())
            continue
    if len(item) != 0:
        item_list.append(item)

    print(item_list[0]["changed_uttrs"])
    print(item_list[0]["original_uttrs"])
    print()
    final_output = []
    for itemIdx, item in enumerate(item_list):
        change_num, org_words, chd_words = (
            item["num_change"],
            item["original_words"],
            item["changed_words"],
        )
        original_uttrs = item["original_uttrs"]
        changed_uttrs = item["changed_uttrs"]

        assert len(org_words) == len(chd_words) == change_num

        if change_uw_to_original:
            context, response = uttr_token.join(original_uttrs[:-1]), original_uttrs[-1]
        else:
            context, response = uttr_token.join(changed_uttrs[:-1]), changed_uttrs[-1]
        context = context.replace(" ##", "")
        response = response.replace(" ##", "")
        context = context.replace("##", "")
        assert "##" not in context
        assert "##" not in response
        final_output.append([context, response])

    return final_output


def get_uw_annotation_legacy(
    fname, change_uw_to_original: bool, replace_golden_to_nota: bool, is_dev
):
    with open(fname, "r") as f:
        ls = [el.strip() for el in f.readlines()]
    item_list, item = [], []
    uttr_token = get_uttr_token()
    nota_token = get_nota_token()
    for line in ls:
        if line == "":
            if len(item) != 0:
                item_list.append(item)
                item = []
            continue

        if "(" in line and ")" in line:
            parsed_tuple = re.findall(r"\([^()]*\)", line)
            num_change = int(line.strip().split()[-1])
            change_map = [make_tuple(el) for el in parsed_tuple]
            assert len(parsed_tuple) == len(change_map)
            item.append(change_map)
            continue
        item.append(line)

    final_output = []
    for item_idx, item in enumerate(item_list):
        change_map, uttrs = item[0], item[1:]
        context = uttr_token.join(uttrs[:-1])
        response = uttrs[-1] if not replace_golden_to_nota else nota_token
        error_case = False
        if change_uw_to_original:
            for change_history in change_map:
                org, chd = change_history
                try:
                    assert chd in context or chd[0].upper() + chd[1:] in context
                except:
                    error_case = True
                    break
                context = context.replace(chd, org).replace(
                    chd[0].upper() + chd[1:], org
                )
        if not error_case:
            final_output.append([context, response])

    if is_dev:
        return final_output[: int(len(final_output) * 0.3)]
    else:
        return final_output[int(len(final_output) * 0.3) :]


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


class RankerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        setname: str,
        max_seq_len: int = 300,
        uttr_token: str = "[UTTR]",
        tensor_fname: str = None,
        corrupted_dataset=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.uttr_token = uttr_token
        self.corrupted_dataset = corrupted_dataset
        assert setname in ["train", "dev", "test"]
        self.triplet_fname = "./data/triplet/triplet_{}.pck".format(setname)
        self.triplet_dataset = self._get_triplet_dataset(raw_dataset)
        if tensor_fname is None:
            self.tensor_fname = "./data/triplet/tensor_{}.pck".format(setname)
        else:
            self.tensor_fname = tensor_fname.format(setname)
        self.feature = self._tensorize_triplet_dataset(corrupted_dataset)

    def __len__(self):
        return len(self.feature[0])

    def __getitem__(self, idx):
        return tuple([el[idx] for el in self.feature])

    def _tensorize_triplet_dataset(self, corrupted_dataset):
        if os.path.exists(self.tensor_fname):
            with open(self.tensor_fname, "rb") as f:
                return pickle.load(f)

        ids, masks, labels = [], [], []
        print("Tensorize...")
        for idx, triple in enumerate(tqdm(self.triplet_dataset)):
            assert len(triple) == 3 and all([isinstance(el, str) for el in triple])
            context, pos_uttr, neg_uttr = triple

            positive_sample = self.tokenizer(
                context,
                text_pair=pos_uttr,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            negative_sample = self.tokenizer(
                context,
                text_pair=neg_uttr,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            ids.extend(positive_sample["input_ids"])
            masks.extend(positive_sample["attention_mask"])
            labels.append(1)
            ids.extend(negative_sample["input_ids"])
            masks.extend(negative_sample["attention_mask"])
            labels.append(0)
        assert len(ids) == len(masks) == len(labels)

        data = torch.stack(ids), torch.stack(masks), torch.tensor(labels)
        with open(self.tensor_fname, "wb") as f:
            pickle.dump(data, f)
        return data

    def _get_triplet_dataset(self, raw_dataset):
        """
        Args:
            raw_dataset (List[List[str]]): List of conversation. Each conversation is list of utterance(str).
        """
        print("Triplet filename: {}".format(self.triplet_fname))
        if os.path.exists(self.triplet_fname):
            print(f"{self.triplet_fname} exist!")
            with open(self.triplet_fname, "rb") as f:
                return pickle.load(f)

        triplet_dataset = self._make_triplet_dataset(raw_dataset)
        os.makedirs(os.path.dirname(self.triplet_fname), exist_ok=True)
        with open(self.triplet_fname, "wb") as f:
            pickle.dump(triplet_dataset, f)
        return triplet_dataset

    def _make_triplet_dataset(self, raw_dataset):
        assert isinstance(raw_dataset, list) and all(
            [isinstance(el, list) for el in raw_dataset]
        )
        print(f"{self.triplet_fname} not exist. Make new file...")
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
        for idx, el in enumerate(dataset):
            while True:
                sampled_random_negative = random.sample(all_responses, 1)[0]
                if sampled_random_negative != el[1]:
                    break
            dataset[idx].append(sampled_random_negative)
        return dataset

    def _slide_conversation(self, conversation):
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
