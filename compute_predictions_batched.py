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

    # Pick two number between 10 and 99

    import random
    number_1 = random.randint(10, 99 + 1)
    
    # Pick another two-digit number that doesn't use any digit from the first number and doesn't have the same digit twice
    number_2 = random.randint(10, 99 + 1)
    while any([str(number_2).count(digit) for digit in str(number_1)]) or len(set(str(number_2))) < 2:
        number_2 = random.randint(10, 99 + 1)

    number_1 = 88
    number_2 = 27

    # Split the digits of the second number
    digit_2a = int(str(number_2)[0])
    digit_2b = int(str(number_2)[1])


    # Compute four numbers:
    # - The sum of number 1 and number 2
    # - The sum of number 1 and number 2 (with the digits reversed for the latter)
    # - The summer of number 1 and the first digit of number 2
    # - The sum of number 1 and the second digit of number 2

    sum_1 = number_1 + number_2
    sum_2 = number_1 + int(f'{digit_2b}{digit_2a}')
    sum_3 = number_1 + digit_2a
    sum_4 = number_1 + digit_2b

    # Take the first digit of each sum and put them in a set
    digits = {str(sum_1)[0], str(sum_2)[0], str(sum_3)[0], str(sum_4)[0]}

    probs = {
        digit: []
        for digit in digits
    }

    import numpy as np

    interpolation_factors = np.linspace(0, 1, 100)

    for interpolation_factor in interpolation_factors:

        result = generator.next_prediction_custom(
            f"The sum of {number_1} and {number_2} is ", 
            str(digit_2a), str(digit_2b), interpolation_factor=interpolation_factor
        )

        result = dict(result)

        for digit in digits:
            if digit in result:
                probs[digit].append(result[digit])
            else:
                probs[digit].append(0)


    #print(result)
    import matplotlib.pyplot as plt

    for digit in digits:
        plt.plot(interpolation_factors, probs[digit], label=digit)
    plt.xlabel('Interpolation factor')
    plt.ylabel('Probability')
    plt.legend()
    plt.title(f'{number_1} + {number_2}')

    plt.savefig('interpolation3.png')


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
