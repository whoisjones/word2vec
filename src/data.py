from torchtext.data import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer

import torch
from torch.utils.data import DataLoader

from functools import partial


max_sequence_length = 512


def get_data_iterator(dataset_name: str, split: str):
    if dataset_name == "WikiText2":
        dataset = WikiText2(split=split)
    elif dataset_name == "WikiText103":
        dataset = WikiText103(split=split)
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    dataset = to_map_style_dataset(dataset)
    return dataset


def collate_cbow(batch, text_pipeline, window_size: int):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.

    Context is represented as N=CBOW_N_WORDS past words
    and N=CBOW_N_WORDS future words.

    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.

    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < window_size * 2 + 1:
            continue

        if max_sequence_length:
            text_tokens_ids = text_tokens_ids[:max_sequence_length]

        for idx in range(len(text_tokens_ids) - window_size * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + window_size * 2 + 1)]
            output = token_id_sequence.pop(window_size)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline, window_size: int):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.

    Context is represented as N=SKIPGRAM_N_WORDS past words
    and N=SKIPGRAM_N_WORDS future words.

    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.

    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < window_size * 2 + 1:
            continue

        if max_sequence_length:
            text_tokens_ids = text_tokens_ids[:max_sequence_length]

        for idx in range(len(text_tokens_ids) - window_size * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + window_size * 2 + 1)]
            input_ = token_id_sequence.pop(window_size)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader(
    model_name: str,
    dataset_name: str,
    split: str,
    batch_size: int,
    window_size: int,
    return_vocab: bool,
):
    dataset = get_data_iterator(dataset_name=dataset_name, split=split)
    tokenizer = get_tokenizer("basic_english", language="en")

    if return_vocab:
        vocab = build_vocab_from_iterator(
            map(tokenizer, dataset), specials=["<unk>"], min_freq=5
        )
        vocab.set_default_index(vocab["<unk>"])
    else:
        vocab = None

    text_pipeline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn, text_pipeline=text_pipeline, window_size=window_size
        ),
    )
    return dataloader, vocab
