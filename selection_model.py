import torch
import torch.nn.init as init
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel


class BertSelect(nn.Module):
    def __init__(self, bert: BertModel):
        super(BertSelect, self).__init__()
        self.bert = bert
        self.linear = torch.nn.Linear(768, 1, bias=False)

    def forward(self, ids, mask):
        output, _ = self.bert(ids, mask, return_dict=False)
        cls_ = output[:, 0]
        return self.linear(cls_)

    def get_attention(self, ids, mask):
        output = self.bert(ids, mask, return_dict=True, output_attentions=True)
        prediction = self.linear(output["last_hidden_state"][:, 0])
        return prediction, output["attentions"]


class BertSelectAuxilary(nn.Module):
    def __init__(self, bert: BertModel):
        super(BertSelectAuxilary, self).__init__()
        self.bert = bert
        self.linear = torch.nn.Linear(768, 1, bias=False)
        self.auxlinear = torch.nn.Linear(768, 1, bias=False)

    def forward(
        self, ids, mask, aux_org_ids, aux_org_mask, aux_corrupt_ids, aux_corrupt_mask
    ):
        output, _ = self.bert(ids, mask, return_dict=False)
        cls_ = output[:, 0]
        selection_output = self.linear(cls_)

        org_output, _ = self.bert(aux_org_ids, aux_org_mask, return_dict=False)
        org_cls_ = org_output[:, 0]
        aux_org_linear_output = self.auxlinear(org_cls_)
        aux_org_output = torch.sigmoid(aux_org_linear_output)

        corrupt_output, _ = self.bert(
            aux_corrupt_ids, aux_corrupt_mask, return_dict=False
        )
        corrupt_cls_ = corrupt_output[:, 0]
        aux_corrupt_linear_output = self.auxlinear(corrupt_cls_)
        aux_corrupt_output = torch.sigmoid(aux_corrupt_linear_output)

        return selection_output, aux_org_output, aux_corrupt_output

    def predict(self, ids, mask):
        output, _ = self.bert(ids, mask, return_dict=False)
        cls_ = output[:, 0]
        selection_output = self.linear(cls_)
        return selection_output

    def get_attention(self, ids, mask):
        output = self.bert(ids, mask, return_dict=True, output_attentions=True)
        prediction = self.linear(output["last_hidden_state"][:, 0])
        return prediction, output["attentions"]
