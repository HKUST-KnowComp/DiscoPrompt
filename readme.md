Code For ACL2023 Paper “DiscoPrompt: Path Prediction Prompt Tuning for Implicit Discourse Relation Recognition”

It contains the following file:
* DiscoPrompt.py
* discoverbalizer.py
* data_utils.py
* utils.py
* requirements.txt

## Prerequisites
* All Environment requirements list in "requirements.txt"

## Usage
1. You should download PDTB2.0 Dataset from "https://catalog.ldc.upenn.edu/LDC2008T05" and CoNLL-2016 datset from "https://www.cs.brandeis.edu/~clp/conll16st/dataset.html".
2. Run Command: "pip install -r requirements.txt".
3. Run following command:
"python /DiscoPrompt/DiscoPrompt.py 
--template_id 0 
--max_steps 30000 
--batch_size 4 
--eval_every_steps 250 
--dataset ji 
--model_name_or_path google/t5-v1_1-large
--result_file ./DiscoPrompt/results/DiscoPromptClassification_PDTB2.txt 
--ckpt_file DiscoPromptClassification_PDTB2 
--prompt_lr 0.3"

