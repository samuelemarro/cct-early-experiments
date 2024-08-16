from pathlib import Path

import matplotlib.pyplot as plt

DIGITS = [str(i) for i in range(10)]


# Use in the rainbow order
COLORS = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:pink', 'tab:purple', 'tab:brown', 'xkcd:burnt siena']
COLORS = {k : v for k, v in zip(DIGITS, COLORS)}
COLORS['other_digits'] = 'tab:gray'
COLORS['other_tokens'] = (26 / 255, 25 / 255, 25 / 255)

def plot_interpolation_curve(interpolation_factors, probs, interest_threshold, title, save_path):
    assert all([digit in DIGITS for digit in probs.keys()])
    # A digit is interesting if at any time it has a probability greater than interest_threshold
    interesting_digits = [digit for digit in DIGITS if max(probs[digit]) > interest_threshold]
    # Sort the digits in ascending order
    interesting_digits = sorted(interesting_digits)

    non_interesting_digits = [digit for digit in DIGITS if digit not in interesting_digits]

    # Compute the sum of all non-interesting probs
    non_interesting_probs = [sum([probs[digit][i] for digit in non_interesting_digits]) for i in range(len(interpolation_factors))]

    print('Plotting graph with interesting digits:', interesting_digits)

    for digit in interesting_digits:
        plt.plot(interpolation_factors, probs[digit], label=digit, color=COLORS[digit])
    plt.plot(interpolation_factors, non_interesting_probs, '--', color=COLORS['other_digits'], label='Other Digits')

    # Compute the remaining probability as 1 - everything else
    remaining_probs = [1 - (sum([probs[digit][i] for digit in DIGITS])) for i in range(len(interpolation_factors))]
    plt.plot(interpolation_factors, remaining_probs, ':', color=COLORS['other_tokens'], label='Other Tokens')

    plt.xlabel('Interpolation Factor ($\\varphi$)')
    plt.ylabel('Probability')
    plt.legend()
    if title is not None:
        plt.title(title)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()