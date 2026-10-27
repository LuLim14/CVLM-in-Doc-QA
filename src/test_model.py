import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

from modeling import CVLM, ModelArguments, TrainingArguments


def test_train_forward(model, input_embeds, prompt_ids, answer_ids):
    output = model(input_embeds, prompt_ids, answer_ids)
    return output["loss"], output["logits"]

if __name__ == "__main__":
    model_args = ModelArguments()
    training_args = TrainingArguments()
    model = CVLM(model_args, training_args)
    model.eval()
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)

    data = np.load(
        "/workspace-SR004.nfs2/acherepanov/compress_project/compress_project/data/dataset_100_samples/embeddings.npy",
        allow_pickle=True,
    ).item()

    saved_indexes = data[
        "indexes"
    ]  # e.g., [0, 2, 5, ...] (skipped rows that were filtered)
    saved_embeddings = data["embeddings"]  # Corresponding vectors

    original_ds = load_dataset("sggetao/PwC", split="train")

    array_pos = 0

    original_db_id = saved_indexes[array_pos]
    record = original_ds[int(original_db_id)]

    input_embeds = saved_embeddings[array_pos]
    input_embeds = torch.from_numpy(input_embeds).to(device=model.encoder.device, dtype=torch.bfloat16)
    input_embeds = input_embeds.unsqueeze(0) # for batch size 1
   

    prompt_ids = tokenizer(record["input"], return_tensors="pt").input_ids.to(device=model.encoder.device)
    answer_ids = tokenizer(record["answer"], return_tensors="pt").input_ids.to(device=model.encoder.device)

    loss, logits = test_train_forward(model, input_embeds, prompt_ids, answer_ids)
    print(f"Loss: {loss}")
    print(f"Logits: {logits}")