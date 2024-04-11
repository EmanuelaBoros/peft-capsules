from .common import *
import random
import pandas as pd

HISTORICAL_PROMPTS = {
    "empty": "",
    "random": "",
    "when": "When did ",
    "when_all_caps": "When did ",
}


# def make_historical_figure_entity_df(


def make_hipe_figure_prompt_dataset(short_prompt, prompt, tokenizer, entity_df):
    # entity_df["occupation"] = entity_df["occupation"].fillna("")
    dataset_strings = []
    for _, row in entity_df.iterrows():
        entity_string = ""
        # if short_prompt.endswith("occupation"):
        #     add_space = len(row["occupation"]) > 0
        #     entity_string += row["occupation"] + (" " if add_space else "")
        entity_string += row["entity_name"]

        if short_prompt.endswith("all_caps"):
            entity_string = entity_string.upper()

        dataset_strings.append(prompt + entity_string)

    token_ids = tokenizer.batch_encode_plus(
        dataset_strings,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    if short_prompt == "random":
        random_prompts = torch.randint(
            low=100,
            high=token_ids.max().item(),
            size=(token_ids.shape[0], 10),
            dtype=torch.long,
        )
        token_ids = torch.cat([random_prompts, token_ids], dim=1)

    # add bos token
    token_ids = torch.cat(
        [
            torch.ones(token_ids.shape[0], 1, dtype=torch.long)
            * tokenizer.bos_token_id,
            token_ids,
        ],
        dim=1,
    )

    prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
    entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
    entity_mask[:, prompt_tokens] = False
    entity_mask[token_ids == tokenizer.pad_token_id] = False

    dataset = datasets.Dataset.from_dict(
        {
            "entity": entity_df.entity_name.values.tolist(),
            "input_ids": token_ids.tolist(),
            "entity_mask": entity_mask.tolist(),
        }
    )

    dataset.set_format(type="torch", columns=["input_ids"])

    return dataset
