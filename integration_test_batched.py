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
    max_number = 49

    number_1 = random.randint(10, max_number + 1)
    
    # Pick another two-digit number that doesn't use any digit from the first number and doesn't have the same digit twice
    number_2 = random.randint(10, max_number + 1)
    while any([str(number_2).count(digit) for digit in str(number_1)]) or len(set(str(number_2))) < 2:
        number_2 = random.randint(10, max_number + 1)

    # number_1 = 88
    # number_2 = 27
    number_1 = 24
    number_2 = 13

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
    digits = {str(x) for x in range(10)} #{str(sum_1)[0], str(sum_2)[0], str(sum_3)[0], str(sum_4)[0]}
    other_digits = set([str(x) for x in range(10)]) - digits

    probs = {
        digit: []
        for digit in digits
    }
    probs_other_digits = []

    import numpy as np

    interpolation_factors = np.linspace(0, 1, 100)

    for interpolation_factor in interpolation_factors:

        tokens = generator.tokenizer.encode("The sum of 24 and 33 is ", bos=True, eos=False)

        # Indexing as [0, 1, 2, 3, 4, 5, 6, 7, 8] + [8, 9, 10, 11] + [11, 12]. Note the repetition of 8 and 11
        # The sum of 24 and [' ', 1, 3, is] is
        new_tokens = tokens[:9] + [tokens[8], tokens[9], tokens[10], tokens[11]] + tokens[11:] 

        #idx = torch.arange(len(tokens), device='cuda')
        idx = torch.tensor(list(range(9)) + [8 + interpolation_factor, 9, 10, 11 - interpolation_factor] + list(range(11, len(tokens))), device='cuda')
        #print(idx)

        h = generator.model.tok_embeddings(torch.tensor(new_tokens, device='cuda').unsqueeze(0))

        #print(tokens)
        #print([generator.tokenizer.decode([t]) for t in tokens])

        # print([
        #     (i.item(), t, generator.tokenizer.decode([t])) for
        #     i, t in zip(idx, new_tokens)
        # ])
        result = generator.next_prediction_given_raw(idx, h)

        result = dict(result)

        for digit in digits:
            if digit in result:
                probs[digit].append(result[digit])
            else:
                probs[digit].append(0)
        
        other_digits_prob = 0
        for other_digit in other_digits:
            if other_digit in result:
                other_digits_prob += result[other_digit]
        probs_other_digits.append(other_digits_prob)


    #print(result)
    import matplotlib.pyplot as plt

    for digit in digits:
        plt.plot(interpolation_factors, probs[digit], label=digit)
    #plt.plot(interpolation_factors, probs_other_digits, label='Other Digits')

    #other_tokens_probs = [1 - (prob_other_digits + sum([probs[digit][i] for digit in digits])) for i, prob_other_digits in enumerate(probs_other_digits)]
    #plt.plot(interpolation_factors, other_tokens_probs, label='Other Tokens')

    plt.xlabel('Interpolation factor')
    plt.ylabel('Probability')
    plt.legend()
    plt.title(f'{number_1} + {number_2}')

    plt.savefig('interpolation5.png')


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
