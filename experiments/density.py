import sys
sys.path.append('.')

import json

import torch

from plot_utils import plot_interpolation_curve
from experiments.common import append_results_info, DIGITS, get_args, load_model

def run(
    ckpt_dir: str,
    tokenizer_path: str
):
    generator = load_model(
        ckpt_dir, tokenizer_path,
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

            interpolation_factors = list(range(1, config['max_density'] + 1))

            tokens = generator.tokenizer.encode(f"The sum of {number_1} and {number_2} is ", bos=True, eos=False)
            assert len(tokens) == 13
            
            embeddings = generator.model.tok_embeddings(torch.tensor(tokens, device='cuda').unsqueeze(0))

            for interpolation_factor in interpolation_factors:
                repeated_embeddings = torch.repeat_interleave(embeddings, interpolation_factor, dim=1)

                # The arange goes from 1 to len(tokens) * interpolation_factor because CCT indices are 1-indexed
                idx = torch.arange(1, len(tokens) * interpolation_factor + 1, device='cuda') / interpolation_factor

                result = generator.next_prediction_given_raw(idx, repeated_embeddings, integration_technique=integration_technique)

                append_results_info(probs, result)

            plot_interpolation_curve(interpolation_factors, probs, interest_threshold, None, f'results/density/{integration_technique}/density_{setup_id}.png')


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path
    )
