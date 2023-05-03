from openprompt.utils.crossfit_metrics import evaluate as crossfit_evaluate
from sklearn.metrics import classification_report #Classification Report
import torch
import json

def evaluate(prompt_model, dataloader, dataset, Processor, use_cuda, num_classes, test = False):
    prompt_model.eval()
    predictions = []
    ground_truths = []

    predictions_4class = []
    ground_truths_4class = []    

    instance_start_number = 0
    instance_number = 0 

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.to(prompt_model.device)

        logits, _, each_logits = prompt_model(inputs)
        predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        
        # instance_number used to get position of current insatnce in dataset
        instance_number = instance_number + len(logits)
        tgt_text = get_labels(dataset, instance_start_number, instance_number,Processor)
        ground_truths.extend(tgt_text)
        instance_start_number = instance_number

        # For 4class label and prediction
        predictions_4class.extend(get_4class_labels(torch.argmax(logits, dim=-1).cpu().tolist(), num_classes)) #obatin 4 class prediction from 11class prediction
        four_class_labels = get_4class_labels(tgt_text, num_classes)
        ground_truths_4class.extend(four_class_labels)

    assert len(predictions)==len(ground_truths), (len(predictions), len(ground_truths))
    assert len(predictions_4class)==len(ground_truths_4class), (len(predictions_4class), len(ground_truths_4class))

    predictions, ground_truths = processing_output_multilabel(predictions,ground_truths)
    predictions_4class, ground_truths_4class = processing_output_multilabel(predictions_4class,ground_truths_4class)

    # shown one example
    print(f"predictions {predictions[0]}, ground_truths {ground_truths[0]}")
    Acc_score = crossfit_evaluate(predictions, ground_truths, metric="ACC")
    F1_score = crossfit_evaluate(predictions, ground_truths, metric="Classification-F1")

    Acc_score_4class = crossfit_evaluate(predictions_4class, ground_truths_4class, metric="ACC")
    F1_score_4class = crossfit_evaluate(predictions_4class, ground_truths_4class, metric="Classification-F1")
    
    if test:
        print("11 Class Classification Report")
        print(classification_report(ground_truths, predictions, digits=4))

        print("4 Class Classification Report")
        print(classification_report(ground_truths_4class, predictions_4class, digits=4))
        
    return Acc_score, F1_score, Acc_score_4class, F1_score_4class

def get_labels(dataset,instance_start_number,instance_number, Processor):
    tgt_text = []
    #Processing the data (a prediction is regraded as correct once it matches one of the ground-truth lables)
    #get the same set of data instance with batch data from dataload 
    for datainstance in dataset[instance_start_number:instance_number]: 
        label_id = get_label_id(datainstance.meta["multi_label"])
        if len(label_id) > 1: 
            tgt_text.extend([label_id])
        else:
            tgt_text.extend(label_id)
    return tgt_text

def get_4class_labels(moreclass_labels, num_classes = 11):
    class4_labels = []
    for label in moreclass_labels:
        if num_classes == 4:
            class4_labels.append(label)
        if num_classes == 11:
            if isinstance(label, list):
                class4_labels.append([mapping_4class_to_11class(each_multi_label)for each_multi_label in label])
            else:
                class4_labels.append(mapping_4class_to_11class(label))
        elif num_classes == 14:
            if isinstance(label, list):
                class4_labels.append([mapping_4class_to_14class(each_multi_label)for each_multi_label in label])
            else:
                class4_labels.append(mapping_4class_to_14class(label))  
        elif num_classes == 15:
            if isinstance(label, list):
                class4_labels.append([mapping_4class_to_15class(each_multi_label)for each_multi_label in label])
            else:
                class4_labels.append(mapping_4class_to_15class(label))  
    return class4_labels


def get_label_id(label_list):
   return [i for i, x in enumerate(label_list) if x == 1]

# for obtain 4 class from 11class 
def mapping_4class_to_11class(label):
    if label < 2:
        class4_label = int(0)
    elif label > 1 and label < 4:
        class4_label = int(1)
    elif label > 3 and label < 9:
        class4_label = int(2)
    elif label > 8 :
        class4_label = int(3)
    return class4_label

# for obtain 4 class from 14class 
def mapping_4class_to_14class(label):
    if label < 2:
        class4_label = int(0)
    elif label > 1 and label < 5:
        class4_label = int(1)
    elif label > 4 and label < 11:
        class4_label = int(2)
    elif label > 10 :
        class4_label = int(3)
    return class4_label

# for obtain 4 class from 15class 
def mapping_4class_to_15class(label):
    if label < 2:
        class4_label = int(0)
    elif label > 1 and label < 5:
        class4_label = int(1)
    elif label > 4 and label < 11:
        class4_label = int(2)
    elif label > 10 and label < 14:
        class4_label = int(3)
    elif label > 13:
        class4_label = int(4)
    return class4_label

def processing_output_multilabel(predictions,ground_truths):
    processed_prediction = []
    processed_ground_truth= []    
    for (prediction, label) in zip(predictions,ground_truths):
        if type(label)==list:
            if prediction in label :
                label = prediction
            else:
                label = label[0]
        processed_prediction.append(prediction)
        processed_ground_truth.append(label)
    return processed_prediction, processed_ground_truth


