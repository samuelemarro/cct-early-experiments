# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

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

    def next_prediction(self, prompt):
        prompt_tokens = torch.tensor([self.tokenizer.encode(prompt, bos=True, eos=False)], device='cuda').long()

        return self.next_prediction_given_tokens(prompt_tokens)
        
    def next_prediction_custom(self, prompt, flip_a, flip_b, interpolation_factor):
        # Runs as normal, but flips th idx_override of two tokens (flip_a and flip_b)
        #print('Prompt:', prompt)
        
        # Convert flip_a and flip_b into ids
        flip_a_ids = self.tokenizer.encode(flip_a, bos=False, eos=False)
        flip_b_ids = self.tokenizer.encode(flip_b, bos=False, eos=False)

        # The first ID is 29871, which marks the beginning of a word
        assert flip_a_ids[0] == 29871
        assert flip_b_ids[0] == 29871
        assert len(flip_a_ids) == 2, 'Flip a should be a single token'
        assert len(flip_b_ids) == 2, 'Flip b should be a single token'

        flip_a_id = flip_a_ids[1]
        flip_b_id = flip_b_ids[1]

        # print('Flip a:', flip_a_id)
        # print('Flip b:', flip_b_id)
        #assert len(flip_a_id) == 1, 'Flip a should be a single token'
        #assert len(flip_b_id) == 1, 'Flip b should be a single token'
        #flip_a_id = flip_a_id[0]
        #flip_b_id = flip_b_id[0]

        prompt_tokens = torch.tensor([self.tokenizer.encode(prompt, bos=True, eos=False)], device='cuda').long()
        idx_override = torch.arange(prompt_tokens.shape[1], device='cuda').long()

        # print('Prompt tokens:', prompt_tokens[0].tolist())
        # Convert explicitly each token to its id
        # print('Decoded prompt:', [self.tokenizer.decode([token]) for token in prompt_tokens[0].tolist()])

        # Find the position of a and b, ensuring that they only appear once
        a_pos = None
        b_pos = None
        for i, token in enumerate(prompt_tokens[0]):
            if token == flip_a_id:
                if a_pos is not None:
                    raise ValueError("flip_a appears multiple times in the prompt")
                a_pos = i
            if token == flip_b_id:
                if b_pos is not None:
                    raise ValueError("flip_b appears multiple times in the prompt")
                b_pos = i
        
        # print('A pos:', a_pos, 'B pos:', b_pos)

        # Flip the idx_override of a and b
        idx_override[a_pos] = b_pos
        idx_override[b_pos] = a_pos

        # print('Idx override:', idx_override.tolist())

        prompt_tokens_with_override = prompt_tokens[:, idx_override]
        # Decode the prompt with the override
        # print('Overriden:', self.tokenizer.decode(prompt_tokens_with_override[0].tolist()))


        return self.next_prediction_given_tokens(prompt_tokens, idx_override=idx_override, interpolation_factor=interpolation_factor)

    def next_prediction_given_tokens(self, tokens, idx_override = None, interpolation_factor = None):
        prev_pos = 0
        cur_pos = len(tokens[0])
        feeded_tokens = tokens[:, prev_pos:cur_pos]
        logits = self.model(feeded_tokens, prev_pos, idx_override=idx_override, interpolation_factor=interpolation_factor)

        probs = torch.softmax(logits, dim=-1)

        # Take the 10 most likely tokens with top-k
        top_k = 10
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Return a list of pairs (decoded token, probability)
        return [(self.tokenizer.decode([idx.item()]), prob.item()) for idx, prob in zip(probs_idx[0][:top_k], probs_sort[0][:top_k])]

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_ids: List[int] = None,
        stop_words: List[str] = None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            i = tokens[:, prev_pos:cur_pos]
            logits = self.model(i, prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            
            if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
                break

        tokens[tokens == self.tokenizer.pad_id] = self.tokenizer.eos_id
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        #print(decoded)
        return [postprocessing(i, stop_words) for i in decoded]


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
