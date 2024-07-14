import sys
sys.path.append('.')

import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from plot_utils import plot_interpolation_curve

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

def cartesian_to_spherical(x):
    # x is a tensor of size (batch_size, n_dim)

    # Compute the radius
    r = torch.norm(x, dim=-1, keepdim=True)

    # Compute the angles
    angles = torch.zeros(x.shape[0], x.shape[1] - 1, device=x.device)

    for i in range(x.shape[1] - 1):
        # The i-th angle is equal to atan2(sqrt(x_n^2 + x_(n+1)^2 + ... + x_(i+1)^2), x_i)

        # Compute the square of the sum of the squares
        sum_of_squares = torch.sum(x[:, i+1:] ** 2, dim=-1)

        # Compute the square root
        sqrt = torch.sqrt(sum_of_squares)

        # Compute the angle
        angles[:, i] = torch.atan2(sqrt, x[:, i])
    
    return torch.cat([r, angles], dim=-1)

def spherical_to_cartesian(theta):
    # theta is a tensor of size (batch_size, n_dim)

    # Extract the radius
    r = theta[:, 0]

    # Extract the angles
    angles = theta[:, 1:]

    # Compute the Cartesian coordinates
    cartesian = torch.zeros(theta.shape[0], theta.shape[1], device=theta.device)

    for i in range(angles.shape[1]): # Going from 0 to n_dim - 2
        # The i-th Cartesian is equal to sin(theta_0) * sin(theta_1) * ... * sin(theta_(i-1)) * cos(theta_i)

        # Compute the product of the sines
        sin_product = torch.prod(torch.sin(angles[:, :i]), dim=-1)

        # Compute the i-th Cartesian
        cartesian[:, i] = sin_product * torch.cos(angles[:, i])

    # The last Cartesian is equal to sin(theta_0) * sin(theta_1) * ... * sin(theta_(n_dim - 1))
    # This is equal to the product of the sines
    cartesian[:, -1] = torch.prod(torch.sin(angles), dim=-1)   

    return r * cartesian

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

    number_1 = 24
    number_2 = 13
    number_2_alternative = 31

    digits = {str(s) for s in range(10)}

    probs = {
        digit: []
        for digit in digits
    }

    import numpy as np

    interpolation_factors = np.linspace(0, 1, 100, endpoint=False)

    def linear_interpolation(original, target, factor):
        return original * (1 - factor) + target * factor

    def spherical_interpolation(original, target, factor):
        original_sphere = cartesian_to_spherical(original.unsqueeze(0))[0]
        target_sphere = cartesian_to_spherical(target.unsqueeze(0))[0]
        interpolated_sphere = original_sphere * (1 - factor) + target_sphere * factor

        return spherical_to_cartesian(interpolated_sphere.unsqueeze(0))[0]

    tokens = generator.tokenizer.encode(f"The sum of {number_1} and {number_2} is ", bos=True, eos=False)
    assert len(tokens) == 13
    
    # 0, ..., 8, 8, 9, 10, 11, 11, 12
    # Note the repetition of 8 and 11
    extended_tokens = tokens[:9] + [tokens[8], tokens[9], tokens[10], tokens[11]] + tokens[11:] 
    extended_embeddings = generator.model.tok_embeddings(torch.tensor(extended_tokens, device='cuda').unsqueeze(0))

    for interpolation_factor in interpolation_factors:
        extended_idx = torch.tensor(list(range(9)) + [8 + interpolation_factor, 9 + interpolation_factor * 0.5, 10 - interpolation_factor * 0.5, 11 - interpolation_factor] + list(range(11, len(tokens))), device='cuda')

        result = generator.next_prediction_given_raw(extended_idx, extended_embeddings)

        result = dict(result)

        for digit in digits:
            if digit in result:
                probs[digit].append(result[digit])
            else:
                probs[digit].append(0)
    
    interest_threshold = 0.05
    
    plot_interpolation_curve(interpolation_factors, probs, interest_threshold, None, 'results/core/time_interpolation.png')


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
