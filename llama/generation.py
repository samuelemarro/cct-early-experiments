# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch

from llama.tokenizer import Tokenizer


class LLaMA:
    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def _should_stop(self, tokens, prompt_tokens, stop_ids, stop_words):
        """credits go to: https://github.com/galatolofederico/vanilla-llama"""
        if stop_ids is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                g = t[len(p):].tolist()
                for stop_id in stop_ids:
                    if stop_id in g:
                        do_stop[i] = True

            if all(do_stop):
                return True

        if stop_words is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                t = t.clone()
                g = t[len(p):]
                g[g == self.tokenizer.pad_id] = self.tokenizer.eos_id
                g = g.tolist()
                d = self.tokenizer.decode(g)
                for stop_word in stop_words:
                    if stop_word in d:
                        do_stop[i] = True

            if all(do_stop):
                return True

        return False

    def next_prediction_given_raw(self, idx, embeddings, integration_technique='riemann', integration_start=0):
        logits = self.model.forward_raw(idx, embeddings, integration_technique=integration_technique, integration_start=integration_start)

        probs = torch.softmax(logits, dim=-1)

        # Sort the probabilities
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Return a list of pairs (decoded token, probability)
        return [(self.tokenizer.decode([idx.item()]), idx.item(), prob.item()) for idx, prob in zip(probs_idx[0], probs_sort[0])]


def postprocessing(output_text, stop_words=None, threshold=10):
    sentences = output_text.split(".")
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > threshold and sentence[-1] == ".":
            filtered_sentences.append(sentence)
    r = '.'.join(sentences).strip()
    if stop_words:
        for w in stop_words:
            if r.endswith(w):
                r = r[0:-len(w)].strip()
    if r[-1] != '.':
        r += '...'
    return r


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
