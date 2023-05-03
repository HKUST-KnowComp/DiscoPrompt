"""
This file contains the logic for loading data for all RelationClassification tasks.
"""

import csv
import json
import os

from typing import *

from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample
from openprompt.utils.logging import logger


CLAUSE_SEPARATOR_SET = set(list(".,:;?!~-"))
CHARACTER_SET = set(map(chr, range(97, 97 + 26))) | set(map(chr, range(65, 65 + 26)))


def post_process_arg1(arg1):
    # capitalize the first character
    i = 0
    while i < len(arg1) and arg1[i] not in CHARACTER_SET:
        i += 1

    # remove separators
    j = len(arg1)
    while j - 1 >= 0 and arg1[j - 1] in CLAUSE_SEPARATOR_SET:
        j -= 1

    k = min(i + 1, j)
    arg1 = arg1[:k].upper() + arg1[k:j]

    return arg1


def post_process_arg2(arg2):
    # lowercase the first character
    i = 0
    while i < len(arg2) and arg2[i] not in CHARACTER_SET:
        i += 1

    # remove separators
    j = len(arg2)
    while j - 1 >= 0 and arg2[j - 1] in CLAUSE_SEPARATOR_SET:
        j -= 1

    k = min(i + 1, j)
    arg2 = arg2[:k].upper() + arg2[k:j]

    return arg2


class PDTB2Processor(DataProcessor):
    rel_map_4 = {
        "Comparison": 0,
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 0,
        "Comparison.Contrast.Juxtaposition": 0,
        "Comparison.Contrast.Opposition": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Pragmatic contrast": 0,
        "Contingency": 1,
        "Contingency.Cause": 1,
        "Contingency.Cause.Reason": 1,
        "Contingency.Cause.Result": 1,
        "Contingency.Condition": 1,
        "Contingency.Condition.Hypothetical": 1,
        "Contingency.Pragmatic cause.Justification": 1,
        "Contingency.Pragmatic condition.Relevance": 1,
        "Expansion": 2,
        "Expansion.Alternative": 2,
        "Expansion.Alternative.Chosen alternative": 2,
        "Expansion.Alternative.Conjunctive": 2,
        "Expansion.Alternative.Disjunctive": 2,
        "Expansion.Conjunction": 2,
        "Expansion.Exception": 2,
        "Expansion.Instantiation": 2,
        "Expansion.List": 2,
        "Expansion.Restatement": 2,
        "Expansion.Restatement.Equivalence": 2,
        "Expansion.Restatement.Generalization": 2,
        "Expansion.Restatement.Specification": 2,
        "Temporal": 3,
        "Temporal.Asynchronous.Precedence": 3,
        "Temporal.Asynchronous.Succession": 3,
        "Temporal.Synchrony": 3
    }
    
    rels_4 = ["Comparison", "Contingency", "Expansion", "Temporal"]

    rel_map_11 = {
        # "Comparison",
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 1,
        "Comparison.Contrast.Juxtaposition": 1,
        "Comparison.Contrast.Opposition": 1,
        # "Comparison.Pragmatic concession",
        # "Comparison.Pragmatic contrast",

        # "Contingency",
        "Contingency.Cause": 2,
        "Contingency.Cause.Reason": 2,
        "Contingency.Cause.Result": 2,
        "Contingency.Pragmatic cause.Justification": 3,
        # "Contingency.Condition",
        # "Contingency.Condition.Hypothetical",
        # "Contingency.Pragmatic condition.Relevance",

        # "Expansion",
        "Expansion.Alternative": 4,
        "Expansion.Alternative.Chosen alternative": 4,
        "Expansion.Alternative.Conjunctive": 4,
        "Expansion.Conjunction": 5,
        "Expansion.Instantiation": 6,
        "Expansion.List": 7,
        "Expansion.Restatement": 8,
        "Expansion.Restatement.Equivalence": 8,
        "Expansion.Restatement.Generalization": 8,
        "Expansion.Restatement.Specification": 8,
        # "Expansion.Alternative.Disjunctive",
        # "Expansion.Exception",

        # "Temporal",
        "Temporal.Asynchronous": 9,
        "Temporal.Asynchronous.Precedence": 9,
        "Temporal.Asynchronous.Succession": 9,
        "Temporal.Synchrony": 10
    }
    rels_11 = [
        "Comparison.Concession", "Comparison.Contrast",
        "Contingency.Cause", "Contingency.Pragmatic cause.Justification",
        "Expansion.Alternative", "Expansion.Conjunction", "Expansion.Instantiation", "Expansion.List", "Expansion.Restatement",
        "Temporal.Asynchronous", "Temporal.Synchrony"
    ]

    def __init__(self, num_labels=4):
        super().__init__()
        if num_labels == 4:
            self._labels = PDTB2Processor.rels_4
            self._label_mapping = PDTB2Processor.rel_map_4
        elif num_labels == 11:
            self._labels = PDTB2Processor.rels_11
            self._label_mapping = PDTB2Processor.rel_map_11
        else:
            raise NotImplementedError

    def get_label_id(self, label) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping.get(label, None)

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        num_labels = self.get_num_labels()
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)

                label = None
                multi_label = [0] * num_labels
                for k in ["Conn1Sense1", "Conn1Sense2", "Conn2Sense1", "Conn2Sense2"]:
                    sense = example_json[k]
                    lbl = self.get_label_id(sense)
                    if lbl is not None:
                        multi_label[lbl] = 1
                        if label is None:
                            label = lbl
                if label is None:
                    continue

#                 if example_json["Arg1SpanList"] < example_json["Arg2SpanList"]:
                if int(example_json["Arg1SpanList"].replace("..",",").split(',')[0]) < int(example_json["Arg2SpanList"].replace("..",",").split(',')[0]):
                    text_a = post_process_arg1(example_json["Arg1RawText"])
                    text_b = post_process_arg2(example_json["Arg2RawText"])
                else:
                    text_a = post_process_arg1(example_json["Arg2RawText"])
                    text_b = post_process_arg2(example_json["Arg1RawText"])

                meta = {
                    "conn1": example_json["Conn1"],
                    "conn2": example_json["Conn2"],
                    "conn1_senses": [example_json["Conn1Sense1"], example_json["Conn1Sense2"]],
                    "conn2_senses": [example_json["Conn2Sense1"], example_json["Conn2Sense2"]],
                    "multi_label": multi_label
                }
                while len(meta["conn1_senses"]) and meta["conn1_senses"][-1] == "":
                    meta["conn1_senses"].pop()
                while len(meta["conn2_senses"]) and meta["conn2_senses"][-1] == "":
                    meta["conn2_senses"].pop()

                guid = "%d_%s_wsj_%02d%02d" % (
                    choicex, example_json["RelationType"], example_json["Section"], example_json["FileNumber"]
                )
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, meta=meta, label=label) #, multi_label=multi_label
                examples.append(example)
        return examples

class PDTB2EXPProcessor(DataProcessor):
    rel_map_4 = {
        "Comparison": 0,
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 0,
        "Comparison.Contrast.Juxtaposition": 0,
        "Comparison.Contrast.Opposition": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Pragmatic contrast": 0,
        "Contingency": 1,
        "Contingency.Cause": 1,
        "Contingency.Cause.Reason": 1,
        "Contingency.Cause.Result": 1,
        "Contingency.Condition": 1,
        "Contingency.Condition.Hypothetical": 1,
        "Contingency.Pragmatic cause.Justification": 1,
        "Contingency.Pragmatic condition.Relevance": 1,
        "Contingency.Condition.Unreal present": 1,  #later join
        "Contingency.Condition.Unreal past": 1, #later join 
        "Contingency.Condition.General": 1,  #later join
        "Contingency.Condition.Factual past": 1,  #later join
        "Contingency.Condition.Factual present": 1, #later join
        "Contingency.Pragmatic condition.Implicit assertion": 1, #later join
        "Expansion": 2,
        "Expansion.Alternative": 2,
        "Expansion.Alternative.Chosen alternative": 2,
        "Expansion.Alternative.Conjunctive": 2,
        "Expansion.Alternative.Disjunctive": 2,
        "Expansion.Conjunction": 2,
        "Expansion.Exception": 2,
        "Expansion.Instantiation": 2,
        "Expansion.List": 2,
        "Expansion.Restatement": 2,
        "Expansion.Restatement.Equivalence": 2,
        "Expansion.Restatement.Generalization": 2,
        "Expansion.Restatement.Specification": 2,
        "Temporal": 3,
        "Temporal.Asynchronous": 3, #later join
        "Temporal.Asynchronous.Precedence": 3,
        "Temporal.Asynchronous.Succession": 3,
        "Temporal.Synchrony": 3
    }
    
    rels_4 = ["Comparison", "Contingency", "Expansion", "Temporal"]

    rel_map_11 = {
        "Comparison": 1,
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        # "Comparison.Pragmatic concession",
        "Comparison.Contrast": 1,
        "Comparison.Contrast.Juxtaposition": 1,
        "Comparison.Contrast.Opposition": 1,
        # "Comparison.Pragmatic contrast",
        # "Contingency",
        "Contingency.Cause": 2,
        "Contingency.Cause.Reason": 2,
        "Contingency.Cause.Result": 2,
        "Contingency.Pragmatic cause.Justification": 3,
#         "Contingency.Condition": 3,
#         "Contingency.Condition.Hypothetical": 3,
#         "Contingency.Pragmatic condition.Relevance": 3,
#         "Contingency.Condition.Unreal present": 3,  #later join
#         "Contingency.Condition.Unreal past": 3, #later join 
#         "Contingency.Condition.General": 3,  #later join
#         "Contingency.Condition.Factual past": 3,  #later join
#         "Contingency.Condition.Factual present": 3, #later join
#         "Contingency.Pragmatic condition.Implicit assertion": 3, #later join
        # "Expansion",
        "Expansion.Alternative": 4,
        "Expansion.Alternative.Chosen alternative": 4,
        "Expansion.Alternative.Conjunctive": 4,
        "Expansion.Alternative.Disjunctive": 4,
        "Expansion.Conjunction": 5,
        "Expansion.Instantiation": 6,
        "Expansion.List": 7,
        "Expansion.Restatement": 8,
        "Expansion.Restatement.Equivalence": 8,
        "Expansion.Restatement.Generalization": 8,
        "Expansion.Restatement.Specification": 8,
        # "Expansion.Exception",
        # "Temporal",
        "Temporal.Asynchronous": 9,
        "Temporal.Asynchronous.Precedence": 9,
        "Temporal.Asynchronous.Succession": 9,
        "Temporal.Synchrony": 10 
    }
    rels_11 = [
        "Comparison.Concession", "Comparison.Contrast",
        "Contingency.Cause", "Contingency.Pragmatic cause.Justification",
        "Expansion.Alternative", "Expansion.Conjunction", "Expansion.Instantiation", "Expansion.List", "Expansion.Restatement",
        "Temporal.Asynchronous", "Temporal.Synchrony"
    ]

    def __init__(self, num_labels=4):
        super().__init__()
        if num_labels == 4:
            self._labels = PDTB2EXPProcessor.rels_4
            self._label_mapping = PDTB2EXPProcessor.rel_map_4
        elif num_labels == 11:
            self._labels = PDTB2EXPProcessor.rels_11
            self._label_mapping = PDTB2EXPProcessor.rel_map_11
        else:
            raise NotImplementedError

    def get_label_id(self, label) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping.get(label, None)

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        num_labels = self.get_num_labels()
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)

                label = None
                multi_label = [0] * num_labels
                for k in ["Conn1Sense1", "Conn1Sense2", "Conn2Sense1", "Conn2Sense2"]:
                    sense = example_json[k]
                    lbl = self.get_label_id(sense)
                    if lbl is not None:
                        multi_label[lbl] = 1
                        if label is None:
                            label = lbl
                if label is None:
                    continue

                if int(example_json["Arg1SpanList"].replace("..",",").split(',')[0]) < int(example_json["Arg2SpanList"].replace("..",",").split(',')[0]):
                    text_a = post_process_arg1(example_json["Arg1RawText"])
                    text_b = post_process_arg2(example_json["Arg2RawText"])
                else:
                    text_a = post_process_arg1(example_json["Arg2RawText"])
                    text_b = post_process_arg2(example_json["Arg1RawText"])

                meta = {
                    "conn1": example_json["Conn1"],
                    "conn2": example_json["Conn2"],
                    "conn1_senses": [example_json["Conn1Sense1"], example_json["Conn1Sense2"]],
                    "conn2_senses": [example_json["Conn2Sense1"], example_json["Conn2Sense2"]],
                    "multi_label": multi_label
                }
                while len(meta["conn1_senses"]) and meta["conn1_senses"][-1] == "":
                    meta["conn1_senses"].pop()
                while len(meta["conn2_senses"]) and meta["conn2_senses"][-1] == "":
                    meta["conn2_senses"].pop()

                guid = "%d_%s_wsj_%02d%02d" % (
                    choicex, example_json["RelationType"], example_json["Section"], example_json["FileNumber"]
                )
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, meta=meta, label=label) #, multi_label=multi_label
                examples.append(example)

        return examples
    
class PDTB3Processor(DataProcessor):
    rel_map_4 = {
        "Comparison.Concession+SpeechAct.Arg2-as-denier+SpeechAct": 0,
        "Comparison.Concession.Arg1-as-denier": 0,
        "Comparison.Concession.Arg2-as-denier": 0,
        "Comparison.Contrast": 0,
        "Comparison.Similarity": 0,
        "Contingency.Cause+Belief.Reason+Belief": 1,
        "Contingency.Cause+Belief.Result+Belief": 1,
        "Contingency.Cause+SpeechAct.Reason+SpeechAct": 1,
        "Contingency.Cause+SpeechAct.Result+SpeechAct": 1,
        "Contingency.Cause.Reason": 1,
        "Contingency.Cause.Result": 1,
        "Contingency.Negative-cause.NegResult": 1,
        "Contingency.Condition+SpeechAct": 1,
        "Contingency.Condition.Arg1-as-cond": 1,
        "Contingency.Condition.Arg2-as-cond": 1,
        "Contingency.Negative-condition.Arg1-as-negCond": 1,
        "Contingency.Negative-condition.Arg2-as-negCond": 1,
        "Contingency.Purpose.Arg1-as-goal": 1,
        "Contingency.Purpose.Arg2-as-goal": 1,
        "Expansion.Conjunction": 2,
        "Expansion.Disjunction": 2,
        "Expansion.Equivalence": 2,
        "Expansion.Exception.Arg1-as-excpt": 2,
        "Expansion.Exception.Arg2-as-excpt": 2,
        "Expansion.Instantiation.Arg1-as-instance": 2,
        "Expansion.Instantiation.Arg2-as-instance": 2,
        "Expansion.Level-of-detail.Arg1-as-detail": 2,
        "Expansion.Level-of-detail.Arg2-as-detail": 2,
        "Expansion.Manner.Arg1-as-manner": 2,
        "Expansion.Manner.Arg2-as-manner": 2,
        "Expansion.Substitution.Arg1-as-subst": 2,
        "Expansion.Substitution.Arg2-as-subst": 2,
        "Temporal.Asynchronous.Precedence": 3,
        "Temporal.Asynchronous.Succession": 3,
        "Temporal.Synchronous": 3
    }
    rels_4 = ["Comparison", "Contingency", "Expansion", "Temporal"]

    rel_map_20 = {
        "Comparison.Concession+SpeechAct.Arg2-as-denier+SpeechAct": 0,
        "Comparison.Concession.Arg1-as-denier": 1,
        "Comparison.Concession.Arg2-as-denier": 1,
        "Comparison.Contrast": 2,
        "Comparison.Similarity": 3,
        "Contingency.Cause+Belief.Reason+Belief": 4,
        "Contingency.Cause+Belief.Result+Belief": 4,
        "Contingency.Cause+SpeechAct.Reason+SpeechAct": 5,
        "Contingency.Cause+SpeechAct.Result+SpeechAct": 5,
        "Contingency.Cause.Reason": 6,
        "Contingency.Cause.Result": 6,
        "Contingency.Negative-cause.NegResult": 6,
        "Contingency.Condition+SpeechAct": 7,
        "Contingency.Condition.Arg1-as-cond": 8,
        "Contingency.Condition.Arg2-as-cond": 8,
        "Contingency.Negative-condition.Arg1-as-negCond": 8,
        "Contingency.Negative-condition.Arg2-as-negCond": 8,
        "Contingency.Purpose.Arg1-as-goal": 9,
        "Contingency.Purpose.Arg2-as-goal": 9,
        "Expansion.Conjunction": 10,
        "Expansion.Disjunction": 11,
        "Expansion.Equivalence": 12,
        "Expansion.Exception.Arg1-as-excpt": 13,
        "Expansion.Exception.Arg2-as-excpt": 13,
        "Expansion.Instantiation.Arg1-as-instance": 14,
        "Expansion.Instantiation.Arg2-as-instance": 14,
        "Expansion.Level-of-detail.Arg1-as-detail": 15,
        "Expansion.Level-of-detail.Arg2-as-detail": 15,
        "Expansion.Manner.Arg1-as-manner": 16,
        "Expansion.Manner.Arg2-as-manner": 16,
        "Expansion.Substitution.Arg1-as-subst": 17,
        "Expansion.Substitution.Arg2-as-subst": 17,
        "Temporal.Asynchronous.Precedence": 18,
        "Temporal.Asynchronous.Succession": 18,
        "Temporal.Synchronous": 19
    }
    rels_20 = [
        "Comparison.Concession+SpeechAct",
        "Comparison.Concession",
        "Comparison.Contrast",
        "Comparison.Similarity",
        "Contingency.Cause+Belief",
        "Contingency.Cause+SpeechAct",
        "Contingency.Cause",
        "Contingency.Condition+SpeechAct",
        "Contingency.Condition",
        "Contingency.Purpose",
        "Expansion.Conjunction",
        "Expansion.Disjunction",
        "Expansion.Equivalence",
        "Expansion.Exception",
        "Expansion.Instantiation",
        "Expansion.Level-of-detail",
        "Expansion.Manner",
        "Expansion.Substitution",
        "Temporal.Asynchronous",
        "Temporal.Synchronous",
    ]
    
    rel_map_14 = {
        "Comparison.Concession.Arg1-as-denier": 0,
        "Comparison.Concession.Arg2-as-denier": 0,
        "Comparison.Contrast": 1,
        "Contingency.Cause+Belief.Reason+Belief": 2,
        "Contingency.Cause+Belief.Result+Belief": 2,
        "Contingency.Cause.Reason": 3,
        "Contingency.Cause.Result": 3,
        "Contingency.Negative-cause.NegResult": 3,
        "Contingency.Condition.Arg1-as-cond": 4,
        "Contingency.Condition.Arg2-as-cond": 4,
        "Contingency.Purpose.Arg1-as-goal": 5,
        "Contingency.Purpose.Arg2-as-goal": 5,
        "Expansion.Conjunction": 6,
        "Expansion.Equivalence": 7,
        "Expansion.Instantiation.Arg1-as-instance": 8,
        "Expansion.Instantiation.Arg2-as-instance": 8,
        "Expansion.Level-of-detail.Arg1-as-detail": 9,
        "Expansion.Level-of-detail.Arg2-as-detail": 9,
        "Expansion.Manner.Arg1-as-manner": 10,
        "Expansion.Manner.Arg2-as-manner": 10,
        "Expansion.Substitution.Arg1-as-subst": 11,
        "Expansion.Substitution.Arg2-as-subst": 11,
        "Temporal.Asynchronous.Precedence": 12,
        "Temporal.Asynchronous.Succession": 12,
        "Temporal.Synchronous": 13
    }
    rels_14 = [
        "Comparison.Concession",
        "Comparison.Contrast",
        "Contingency.Cause+Belief",
        "Contingency.Cause",
        "Contingency.Condition",
        "Contingency.Purpose",
        "Expansion.Conjunction",
        "Expansion.Equivalence",
        "Expansion.Instantiation",
        "Expansion.Level-of-detail",
        "Expansion.Manner",
        "Expansion.Substitution",
        "Temporal.Asynchronous",
        "Temporal.Synchronous",
    ]
    
    
    def __init__(self, num_labels=4):
        super().__init__()
        if num_labels == 4:
            self._labels = PDTB3Processor.rels_4
            self._label_mapping = PDTB3Processor.rel_map_4
        elif num_labels == 14:
            self._labels = PDTB3Processor.rels_14
            self._label_mapping = PDTB3Processor.rel_map_14
        elif num_labels == 20:
            self._labels = PDTB3Processor.rels_20
            self._label_mapping = PDTB3Processor.rel_map_20
        else:
            raise NotImplementedError

    def get_label_id(self, label) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping.get(label, None)

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        num_labels = self.get_num_labels()
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                if example_json['RelationType'] != "Implicit":
                    continue

                label = None
                multi_label = [0] * num_labels
                for k in ["Conn1Sense1", "Conn1Sense2", "Conn2Sense1", "Conn2Sense2"]:
                    sense = example_json[k]
                    lbl = self.get_label_id(sense)
                    if lbl is not None:
                        multi_label[lbl] = 1
                        if label is None:
                            label = lbl
                if label is None:
                    continue
                
                if int(example_json["Arg1SpanList"].replace("..",",").split(',')[0]) < int(example_json["Arg2SpanList"].replace("..",",").split(',')[0]):
                    text_a = post_process_arg1(example_json["Arg1RawText"])
                    text_b = post_process_arg2(example_json["Arg2RawText"])
                else:
                    text_a = post_process_arg1(example_json["Arg2RawText"])
                    text_b = post_process_arg2(example_json["Arg1RawText"])

                meta = {
                    "conn1": example_json["Conn1"],
                    "conn2": example_json["Conn2"],
                    "conn1_senses": [example_json["Conn1Sense1"], example_json["Conn1Sense2"]],
                    "conn2_senses": [example_json["Conn2Sense1"], example_json["Conn2Sense2"]],
                    "multi_label": multi_label
                }
                while len(meta["conn1_senses"]) and meta["conn1_senses"][-1] == "":
                    meta["conn1_senses"].pop()
                while len(meta["conn2_senses"]) and meta["conn2_senses"][-1] == "":
                    meta["conn2_senses"].pop()

                guid = "%d_%s_%s" % (choicex, example_json["RelationType"], example_json["Filename"])
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, meta=meta, label=label) #, multi_label=multi_label
                examples.append(example)
        return examples


class CoNLL15Processor(DataProcessor):
    rel_map_4 = {
        "Comparison": 0,
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 0,
        "Comparison.Contrast.Juxtaposition": 0,
        "Comparison.Contrast.Opposition": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Pragmatic contrast": 0,
        "Contingency": 1,
        "Contingency.Cause": 1,
        "Contingency.Cause.Reason": 1,
        "Contingency.Cause.Result": 1,
        "Contingency.Condition": 1,
        "Contingency.Condition.Hypothetical": 1,
        "Contingency.Pragmatic cause.Justification": 1,
        "Contingency.Pragmatic condition.Relevance": 1,
        "Expansion": 2,
        "Expansion.Alternative": 2,
        "Expansion.Alternative.Chosen alternative": 2,
        "Expansion.Alternative.Conjunctive": 2,
        "Expansion.Alternative.Disjunctive": 2,
        "Expansion.Conjunction": 2,
        "Expansion.Exception": 2,
        "Expansion.Instantiation": 2,
        "Expansion.List": 2,
        "Expansion.Restatement": 2,
        "Expansion.Restatement.Equivalence": 2,
        "Expansion.Restatement.Generalization": 2,
        "Expansion.Restatement.Specification": 2,
        "Temporal": 3,
        "Temporal.Asynchronous.Precedence": 3,
        "Temporal.Asynchronous.Succession": 3,
        "Temporal.Synchrony": 3
    }
    rels_4 = ["Comparison", "Contingency", "Expansion", "Temporal"]

    rel_map_14 = {
        # "Comparison"
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Contrast": 1,
        "Comparison.Contrast.Juxtaposition": 1,
        "Comparison.Contrast.Opposition": 1,
        "Comparison.Pragmatic contrast": 1,
        # "Contingency"
        "Contingency.Cause.Reason": 2,
        "Contingency.Pragmatic cause.Justification": 2,
        "Contingency.Cause.Result": 3,
        "Contingency.Condition": 4,
        "Contingency.Condition.Hypothetical": 4,
        "Contingency.Pragmatic condition.Relevance": 4,
        # "Contingency.Cause",
        # "Expansion"
        "Expansion.Alternative": 5,
        "Expansion.Alternative.Conjunctive": 5,
        "Expansion.Alternative.Disjunctive": 5,
        "Expansion.Alternative.Chosen alternative": 6,
        "Expansion.Conjunction": 7,
        "Expansion.List": 7,
        "Expansion.Exception": 8,
        "Expansion.Instantiation": 9,
        "Expansion.Restatement": 10,
        "Expansion.Restatement.Equivalence": 10,
        "Expansion.Restatement.Generalization": 10,
        "Expansion.Restatement.Specification": 10,
        # "Temporal",
        "Temporal.Asynchronous.Precedence": 11,
        "Temporal.Asynchronous.Succession": 12,
        "Temporal.Synchrony": 13
    }
    rels_14 = [
        "Comparison.Concession", "Comparison.Contrast", "Contingency.Cause.Reason", "Contingency.Cause.Result",
        "Contingency.Condition", "Expansion.Alternative", "Expansion.Alternative.Chosen alternative",
        "Expansion.Conjunction", "Expansion.Exception", "Expansion.Instantiation", "Expansion.Restatement",
        "Temporal.Asynchronous.Precedence", "Temporal.Asynchronous.Succession", "Temporal.Synchrony"
    ]

    def __init__(self, num_labels=4):
        super().__init__()
        if num_labels == 4:
            self._labels = CoNLL15Processor.rels_4
            self._label_mapping = CoNLL15Processor.rel_map_4
        elif num_labels == 14:
            self._labels = CoNLL15Processor.rels_14
            self._label_mapping = CoNLL15Processor.rel_map_14
        else:
            raise NotImplementedError

    def get_label_id(self, label) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping.get(label, None)

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        num_labels = self.get_num_labels()
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                
                # filter all the relation type not belongs to be Implicit (i.e. {'Explicit': 556, 'EntRel': 200, 'Implicit': 425, 'AltLex': 28})
                if example_json['Type'] != "Implicit":
                    continue

                label = None
                multi_label = [0] * num_labels
                for sense in example_json["Sense"]:
                    lbl = self.get_label_id(sense)
                    if lbl is not None:
                        multi_label[lbl] = 1
                        if label is None:
                            label = lbl
                if label is None:
                    continue

                if example_json["Arg1"]["CharacterSpanList"][0][0] < example_json["Arg2"]["CharacterSpanList"][0][0]:
                    text_a = post_process_arg1(example_json["Arg1"]["RawText"])
                    text_b = post_process_arg2(example_json["Arg2"]["RawText"])
                else:
                    text_a = post_process_arg1(example_json["Arg2"]["RawText"])
                    text_b = post_process_arg2(example_json["Arg1"]["RawText"])

                meta = {
#                     "arg1": example_json["Arg1"]["RawText"],
#                     "arg2": example_json["Arg2"]["RawText"],
                    "conn1": example_json["Connective"]["RawText"],
                    "conn2": "",
                    "conn1_senses": example_json["Sense"],
                    "conn2_senses": [],
                    "multi_label": multi_label
                }
                while len(meta["conn1_senses"]) and meta["conn1_senses"][-1] == "":
                    meta["conn1_senses"].pop()

                guid = "%d_%s_%s" % (choicex, example_json["Type"], example_json["DocID"])
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, meta=meta, label=label)#, multi_label=multi_label)
                examples.append(example)
        return examples

class CoNLL15Processor_ent(DataProcessor):
    rel_map_5 = {
        "Comparison": 0,
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 0,
        "Comparison.Contrast.Juxtaposition": 0,
        "Comparison.Contrast.Opposition": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Pragmatic contrast": 0,
        "Contingency": 1,
        "Contingency.Cause": 1,
        "Contingency.Cause.Reason": 1,
        "Contingency.Cause.Result": 1,
        "Contingency.Condition": 1,
        "Contingency.Condition.Hypothetical": 1,
        "Contingency.Pragmatic cause.Justification": 1,
        "Contingency.Pragmatic condition.Relevance": 1,
        "Expansion": 2,
        "Expansion.Alternative": 2,
        "Expansion.Alternative.Chosen alternative": 2,
        "Expansion.Alternative.Conjunctive": 2,
        "Expansion.Alternative.Disjunctive": 2,
        "Expansion.Conjunction": 2,
        "Expansion.Exception": 2,
        "Expansion.Instantiation": 2,
        "Expansion.List": 2,
        "Expansion.Restatement": 2,
        "Expansion.Restatement.Equivalence": 2,
        "Expansion.Restatement.Generalization": 2,
        "Expansion.Restatement.Specification": 2,
        "Temporal": 3,
        "Temporal.Asynchronous.Precedence": 3,
        "Temporal.Asynchronous.Succession": 3,
        "Temporal.Synchrony": 3,
        "EntRel":4
    }
    rels_5 = ["Comparison", "Contingency", "Expansion", "Temporal", "EntRel"]

    rel_map_15 = {
        # "Comparison"
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Contrast": 1,
        "Comparison.Contrast.Juxtaposition": 1,
        "Comparison.Contrast.Opposition": 1,
        "Comparison.Pragmatic contrast": 1,
        # "Contingency"
        "Contingency.Cause.Reason": 2,
        "Contingency.Pragmatic cause.Justification": 2,
        "Contingency.Cause.Result": 3,
        "Contingency.Condition": 4,
        "Contingency.Condition.Hypothetical": 4,
        "Contingency.Pragmatic condition.Relevance": 4,
        # "Contingency.Cause",
        # "Expansion"
        "Expansion.Alternative": 5,
        "Expansion.Alternative.Conjunctive": 5,
        "Expansion.Alternative.Disjunctive": 5,
        "Expansion.Alternative.Chosen alternative": 6,
        "Expansion.Conjunction": 7,
        "Expansion.List": 7,
        "Expansion.Exception": 8,
        "Expansion.Instantiation": 9,
        "Expansion.Restatement": 10,
        "Expansion.Restatement.Equivalence": 10,
        "Expansion.Restatement.Generalization": 10,
        "Expansion.Restatement.Specification": 10,
        # "Temporal",
        "Temporal.Asynchronous.Precedence": 11,
        "Temporal.Asynchronous.Succession": 12,
        "Temporal.Synchrony": 13,
        "EntRel": 14
    }
    rels_15 = [
        "Comparison.Concession", "Comparison.Contrast", "Contingency.Cause.Reason", "Contingency.Cause.Result",
        "Contingency.Condition", "Expansion.Alternative", "Expansion.Alternative.Chosen alternative",
        "Expansion.Conjunction", "Expansion.Exception", "Expansion.Instantiation", "Expansion.Restatement",
        "Temporal.Asynchronous.Precedence", "Temporal.Asynchronous.Succession", "Temporal.Synchrony",
        "EntRel"
    ]

    def __init__(self, num_labels=5):
        super().__init__()
        if num_labels == 5:
            self._labels = CoNLL15Processor_ent.rels_5
            self._label_mapping = CoNLL15Processor_ent.rel_map_5
        elif num_labels == 15:
            self._labels = CoNLL15Processor_ent.rels_15
            self._label_mapping = CoNLL15Processor_ent.rel_map_15
        else:
            raise NotImplementedError

    def get_label_id(self, label) -> int:
        """get label id of the corresponding label

        Args:
            label: label in dataset

        Returns:
            int: the index of label
        """
        return self.label_mapping.get(label, None)

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        num_labels = self.get_num_labels()
        with open(path, encoding='utf8') as f:
            for choicex, line in enumerate(f):
                example_json = json.loads(line)
                
                # filter all the relation type not belongs to be Implicit (i.e. {'Explicit': 556, 'EntRel': 200, 'Implicit': 425, 'AltLex': 28})
                if example_json['Type'] == "Implicit" or example_json['Type'] == "EntRel":
                    label = None
                    multi_label = [0] * num_labels
                    for sense in example_json["Sense"]:
                        lbl = self.get_label_id(sense)
                        if lbl is not None:
                            multi_label[lbl] = 1
                            if label is None:
                                label = lbl
                    if label is None:
                        continue

                    if example_json["Arg1"]["CharacterSpanList"][0][0] < example_json["Arg2"]["CharacterSpanList"][0][0]:
                        text_a = post_process_arg1(example_json["Arg1"]["RawText"])
                        text_b = post_process_arg2(example_json["Arg2"]["RawText"])
                    else:
                        text_a = post_process_arg1(example_json["Arg2"]["RawText"])
                        text_b = post_process_arg2(example_json["Arg1"]["RawText"])

                    meta = {
    #                     "arg1": example_json["Arg1"]["RawText"],
    #                     "arg2": example_json["Arg2"]["RawText"],
                        "conn1": example_json["Connective"]["RawText"],
                        "conn2": "",
                        "conn1_senses": example_json["Sense"],
                        "conn2_senses": [],
                        "multi_label": multi_label
                    }
                    while len(meta["conn1_senses"]) and meta["conn1_senses"][-1] == "":
                        meta["conn1_senses"].pop()

                    guid = "%d_%s_%s" % (choicex, example_json["Type"], example_json["DocID"])
                    example = InputExample(guid=guid, text_a=text_a, text_b=text_b, meta=meta, label=label)#, multi_label=multi_label)
                    examples.append(example)
        return examples

PROCESSORS = {"pdtb2": PDTB2Processor, "pdtb2_exp":PDTB2EXPProcessor, "pdtb3": PDTB3Processor, "conll15": CoNLL15Processor, "conll15-ent":CoNLL15Processor_ent}
