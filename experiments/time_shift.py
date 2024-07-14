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

    with open('config.json', 'r') as f:
        config = json.load(f)

    interest_threshold = config['interest_threshold']

    for integration_technique in config['integration_techniques']:
        for setup_id, setup in enumerate(config['setups']):
            number_1 = setup['number_1']
            number_2 = setup['number_2']

            digits = {str(s) for s in range(10)}

            probs = {
                digit: []
                for digit in digits
            }

            import numpy as np

            interpolation_factors = np.linspace(0, 1, 100)

            tokens = generator.tokenizer.encode(f"The sum of {number_1} and {number_2} is ", bos=True, eos=False)
            embeddings = generator.model.tok_embeddings(torch.tensor(tokens, device='cuda'))
            idx = torch.arange(embeddings.shape[0], device='cuda')
            # CCTs are 1-indexed
            idx = idx + 1
            
            for interpolation_factor in interpolation_factors:
                actual_idx = idx + interpolation_factor

                # interpolation_start ensures that the first token doesn't have an excessive weight
                result = generator.next_prediction_given_raw(actual_idx, embeddings.unsqueeze(0), integration_technique=integration_technique, integration_start=interpolation_factor)
                #print(result)

                # For some God-forsaken reason, there are multiple results for the same digit. Probably due to equivalent Unicode characters.
                for digit in digits:
                    total = 0
                    for key, value in result:
                        if key == digit:
                            total += value
                    probs[digit].append(total)
            
            plot_interpolation_curve(interpolation_factors, probs, interest_threshold, None, f'results/time_shift/{integration_technique}/time_shift_{setup_id}.png')


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
