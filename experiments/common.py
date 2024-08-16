import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="./llama_data/tokenizer.model"
    )
    return parser.parse_args()

LOCAL_RANK = 0
WORLD_SIZE = 1
MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 1

def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert WORLD_SIZE == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {WORLD_SIZE}"
    ckpt_path = checkpoints[LOCAL_RANK]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=MAX_SEQ_LEN, max_batch_size=MAX_BATCH_SIZE, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)

    return generator

DIGITS = {str(s) for s in range(10)}

def append_results_info(probs, result):
    assert set(DIGITS) == set(probs.keys())

    # For some God-forsaken reason, there are multiple results for the same digit.
    # This is probably due to equivalent Unicode characters. The impact is minimal
    # (on the scale of 1e-12), but for completeness we sum them up

    for digit in DIGITS:
        total = 0
        for key, token_idx, value in result:
            if key == digit:
                total += value
        assert total != 0
        probs[digit].append(total)