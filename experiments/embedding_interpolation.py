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
            number_2_alternative = int(str(number_2)[::-1])

            probs = {
                digit: []
                for digit in DIGITS
            }

            interpolation_factors = np.linspace(0, 1, 100)

            def linear_interpolation(original, target, factor):
                return original * (1 - factor) + target * factor

            tokens_with_standard = generator.tokenizer.encode(f"The sum of {number_1} and {number_2} is ", bos=True, eos=False)
            tokens_with_alternative = generator.tokenizer.encode(f"The sum of {number_1} and {number_2_alternative} is ", bos=True, eos=False)

            standard_embeddings = generator.model.tok_embeddings(torch.tensor(tokens_with_standard, device='cuda'))
            alternative_embeddings = generator.model.tok_embeddings(torch.tensor(tokens_with_alternative, device='cuda'))

            assert standard_embeddings.shape == alternative_embeddings.shape == (13, 4096)

            idx = torch.arange(standard_embeddings.shape[0], device='cuda')

            # CCT indices are 1-indexed
            idx = idx + 1

            for interpolation_factor in interpolation_factors:
                different_indices = [i for i in range(len(standard_embeddings)) if torch.any(standard_embeddings[i] != alternative_embeddings[i])]

                assert len(different_indices) == len(str(number_2)) == len(str(number_2_alternative))

                interpolated_embeddings = standard_embeddings.clone()
                for i in different_indices:
                    interpolated_embeddings[i] = linear_interpolation(standard_embeddings[i], alternative_embeddings[i], interpolation_factor)
                
                interpolated_embeddings = interpolated_embeddings.unsqueeze(0)

                result = generator.next_prediction_given_raw(idx, interpolated_embeddings, integration_technique=integration_technique)

                append_results_info(probs, result)
            
            plot_interpolation_curve(interpolation_factors, probs, interest_threshold, None, f'results/embedding_interpolation/{integration_technique}/embedding_interpolation_{setup_id}.png')


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path
    )
