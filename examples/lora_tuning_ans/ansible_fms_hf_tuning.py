from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


data_file_path = "" # path to train data

### ansible dataset
dataset_ans = load_dataset("json", data_files= data_file_path)

print(dataset_ans)
print(f"features: \n {dataset_ans['train'].features}")


dataset = dataset_ans["train"].map(
     lambda x: {"output_": f"### Input: {x['input']}\n\n### Response: {x['output']}"},
     remove_columns=['input', 'output']
 )

dataset = dataset.rename_column("output_", "output")

dataset.to_json("containers_infra-ent-train-fms.json")

## dataset
# dataset_name = "twitter_complaints"
# dataset = load_dataset("ought/raft", dataset_name)
# classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
# dataset = dataset.map(
#     lambda x: {"text_label": [classes[label] for label in x["Label"]]},
#     batched=True,
#     num_proc=1,
# )

# #make the dataset sfttrainer style

# dataset = dataset["train"].map(
#     lambda x: {"output": f"### Text: {x['Tweet text']}\n\n### Label: {x['text_label']}"},
# )
# dataset.to_json("twitter_complaints.json")


