# for PTRverbalizer
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import *
from openprompt.data_utils import InputFeatures
import torch
from torch import nn
from openprompt.prompts import PTRVerbalizer
from openprompt.prompts import One2oneVerbalizer

class DiscoVerbalizer(PTRVerbalizer):
    """
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        classes (:obj:`Sequence[str]`): A sequence of classes that need to be projected.
        label_words (:obj:`Union[Sequence[Sequence[str]], Mapping[str, Sequence[str]]]`, optional): The label words that are projected by the labels.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 classes: Sequence[str] = None,
                 num_classes: Optional[int] = None,
                 label_words: Optional[Union[Sequence[Sequence[str]], Mapping[str, Sequence[str]]]] = None,
                ):
        super().__init__(tokenizer = tokenizer, classes = classes, num_classes = num_classes)
        self.label_words = label_words

    def on_label_words_set(self):
        super().on_label_words_set()

        self.num_masks = len(self.label_words[0])
        for words in self.label_words:
            if len(words) != self.num_masks:
                raise ValueError("number of mask tokens for different classes are not consistent")
        
        # self.sub_labels = [
        #     list(set([words[i] for words in self.label_words]))
        #     for i in range(self.num_masks)
        # ] # [num_masks, label_size of the corresponding mask]

        # Since orginal self.sub_labels have a bug on the 'set([words[i] for words in self.label_words])'
        # This bug (set()function) will change the order of label words (in our case, it is second logits/label)
        pre_sub_labels = [ #separate label list by masks
            [words[i] for words in self.label_words]
            for i in range(self.num_masks)
        ]
        self.sub_labels = [
            sorted(set(pre_sub_labels[i]), key=pre_sub_labels[i].index) for i in range (len(pre_sub_labels))
        ] # [num_masks, label_size of the corresponding mask]

        self.verbalizers = nn.ModuleList([
            One2oneVerbalizer(tokenizer=self.tokenizer, label_words=labels, post_log_softmax = False)
            for labels in self.sub_labels
        ]) # [num_masks]

        self.label_mappings = nn.Parameter(torch.LongTensor([
            [labels.index(words[j]) for words in self.label_words]
            for j, labels in enumerate(self.sub_labels)
        ]), requires_grad=False) # [num_masks, label_size of the whole task]

    def process_logits(self,
                       logits: torch.Tensor, # [batch_size, num_masks, vocab_size]
                       batch: Union[Dict, InputFeatures],
                       **kwargs):
        each_logits = [ # logits of each verbalizer
            self.verbalizers[i].process_logits(logits = logits[:, i, :], batch = batch, **kwargs)
            for i in range(self.num_masks)
        ] # num_masks * [batch_size, label_size of the corresponding mask]

        label_logits = [
            logits[:, self.label_mappings[j]]
            for j, logits in enumerate(each_logits)
        ]

        logsoftmax = nn.functional.log_softmax(sum(label_logits), dim=-1)

        if 'label' in batch: # TODO not an elegant solution
            each_logsoftmax = [ # (logits of each label) of each mask
                nn.functional.log_softmax(logits, dim=-1)[:, self.label_mappings[j]]
                for j, logits in enumerate(each_logits)
            ] # num_masks * [batch_size, label_size of the whole task]

            return logsoftmax + sum(each_logsoftmax) / len(each_logits), each_logsoftmax, each_logits  # [batch_size, label_size of the whole task]

        return logsoftmax