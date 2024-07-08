import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator


def run(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    local_rank = 0
    world_size = 1
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    generator.model.eval()

    #result = generator.next_prediction("The sum of 24 and 13 is ")

    prob_3 = []
    prob_5 = []

    prob_2 = []
    import numpy as np

    interpolation_factors = np.linspace(0, 1, 100)

    for interpolation_factor in interpolation_factors:

        result = generator.next_prediction_custom("The sum of 24 and 13 is ", '1', '3', interpolation_factor=interpolation_factor)
        prob_3.append([x[1] for x in result if x[0] == '3'][0])
        prob_5.append([x[1] for x in result if x[0] == '5'][0])
        prob_2.append([x[1] for x in result if x[0] == '2'][0])
    #print(result)
    import matplotlib.pyplot as plt
    plt.plot(interpolation_factors, prob_3, label='3')
    plt.plot(interpolation_factors, prob_5, label='5')
    plt.plot(interpolation_factors, prob_2, label='2')
    plt.xlabel('Interpolation factor')
    plt.ylabel('Probability')
    plt.legend()

    plt.savefig('interpolation2.png')


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="./llama_data/tokenizer.model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=0,
        top_p=0.95,
        max_seq_len=1024,
        max_batch_size=1,
    )
