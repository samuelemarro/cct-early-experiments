import sys
sys.path.append('.')

import json

import numpy as np
import torch

from plot_utils import plot_interpolation_curve
from experiments.common import append_results_info, DIGITS, get_args, load_model

def run(
    ckpt_dir: str,
    tokenizer_path: str
):
    generator = load_model(
        ckpt_dir, tokenizer_path
    )
    generator.model.eval()

    with open('config.json', 'r') as f:
        config = json.load(f)

    interest_threshold = config['interest_threshold']

    for integration_technique in config['integration_techniques']:
        for setup_id, setup in enumerate(config['setups']):
            number_1 = setup['number_1']
            number_2 = setup['number_2']

            probs = {
                digit: []
                for digit in DIGITS
            }

            interpolation_factors = np.linspace(config['min_time_scale'], 1, 100)

            tokens = generator.tokenizer.encode(f"The sum of {number_1} and {number_2} is ", bos=True, eos=False)
            embeddings = generator.model.tok_embeddings(torch.tensor(tokens, device='cuda'))
            idx = torch.arange(embeddings.shape[0], device='cuda')

            # CCT indices are 1-indexed
            idx = idx + 1
            
            for interpolation_factor in interpolation_factors:
                actual_idx = idx * interpolation_factor

                result = generator.next_prediction_given_raw(actual_idx, embeddings.unsqueeze(0), integration_technique=integration_technique)
                
                append_results_info(probs, result)
            
            plot_interpolation_curve(interpolation_factors, probs, interest_threshold, None, f'results/time_scale/{integration_technique}/time_scale_{setup_id}.png')


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path
    )
