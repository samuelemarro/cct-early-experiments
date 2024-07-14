from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
import hiq


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 1
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

# Computes the values on the fly, accepting potentially non-integer indices
def compute_freqs_cis(t : torch.Tensor, dim: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=t.device)[: (dim // 2)].float() / dim))
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // 1
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )
        # self.cache_k = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # )
        # self.cache_v = torch.zeros(
        #     (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        # )
        # if hiq.get_env_bool("KV_CAHCHE_IN_GPU", True):
        #     self.cache_k = self.cache_k.cuda()
        #     self.cache_v = self.cache_v.cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        assert start_pos == 0
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # We've disabled the cache. This makes the model stateless, but also slower.
        # It also slightly breaks compatibility with the standard generate function
        # (which feeds inputs one token at the time, and expects the model to remember the previous states)

        #self.cache_k = self.cache_k.to(xq)
        #self.cache_v = self.cache_v.to(xq)

        #self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        #self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        #keys = self.cache_k[:bsz, : start_pos + seqlen]
        #values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # self.freqs_cis = precompute_freqs_cis(
        #     self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        # )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, idx_override = None, interpolation_factor=None, integration_technique='trapezoidal', interpolate_embeddings=True):
        _bsz, seqlen = tokens.shape
        if interpolation_factor is None or not interpolate_embeddings:
            h = self.tok_embeddings(tokens)
        else:
            #print(idx_override)
            actual_interpolation_factor = interpolation_factor if interpolation_factor < 0.5 else 1 - interpolation_factor
            h = actual_interpolation_factor * self.tok_embeddings(tokens)[:, idx_override] + (1 - actual_interpolation_factor) * self.tok_embeddings(tokens)
        #self.freqs_cis = self.freqs_cis.to(h.device)
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        #idx_override = None

        standard_idx = torch.arange(start_pos, start_pos + seqlen, device=h.device)

        # Compute on the fly
        if idx_override is not None:
            #print('Overridden!')
            idx = standard_idx.clone().to(torch.float32)
            idx[idx_override != standard_idx] = ((interpolation_factor * idx_override.to(h.device)) + ((1 - interpolation_factor) * standard_idx))[idx_override != standard_idx].to(torch.float32)
        else:
            idx = standard_idx
        
        return self.forward_raw(idx, h, start_pos, integration_technique)

    def forward_raw(self, idx: torch.Tensor, h: torch.Tensor, start_pos : int, integration_technique='trapezoidal', integration_start=0):
        #idx = standard_idx
        seqlen = h.shape[1]

        assert start_pos == 0

        #print('Idx:', idx.tolist())
        freqs_cis = compute_freqs_cis(idx, self.params.dim // self.params.n_heads)
        #print('Freqs:', freqs_cis.sum(-1).cpu().numpy())

        if seqlen > 1:
            # Compute the multiplicative mask, instead of additive
            multiplicative_mask = torch.zeros((1, 1, seqlen, seqlen), device=h.device)
            # multiplicative_mask[i, j] = 1 if j is used to predict i, 0 otherwise
            # Note: with the current implementation, it's also possible to have other values

            for i in range(seqlen):
                if integration_technique == 'riemann':
                    for j in range(0, seqlen):
                        source_position = idx[j] # The position we are looking at
                        target_position = idx[i] # The position we are trying to predict

                        if target_position < source_position:
                            # We can't look into the future
                            multiplicative_mask[0, 0, i, j] = 0
                            continue

                        # Rectangle integration: the weight of source_position is equal
                        # to how much "time" (delta_x) has passed since the last time
                        # we saw another position (previous_position)

                        all_previous_positions = idx[idx < source_position]

                        if len(all_previous_positions) == 0:
                            previous_position = integration_start
                        else:
                            previous_position = torch.max(all_previous_positions)
                        #print('Target:', target_position, 'Previous:', previous_position)

                        delta_x = source_position - previous_position

                        multiplicative_mask[0, 0, i, j] = delta_x
                elif integration_technique == 'trapezoidal':
                    # TODO: Doesn't take into account integration_start
                    assert False
                    for j in range(0, seqlen):
                        source_position = idx[j] # The position we are looking at
                        target_position = idx[i] # The position we are trying to predict

                        if target_position < source_position:
                            # We can't look into the future
                            multiplicative_mask[0, 0, i, j] = 0
                            continue

                        # Riemann integration: the weight of source_position is equal
                        # to how much "time" (delta_x) has passed since the last time
                        # we saw another position (previous_position)

                        all_previous_positions = idx[idx < source_position]
                        all_next_positions = idx[idx > source_position]

                        previous_position = torch.max(all_previous_positions) if len(all_previous_positions) > 0 else None
                        next_position = torch.min(all_next_positions) if len(all_next_positions) > 0 else None
                        #print('Target:', target_position, 'Previous:', previous_position)

                        contribution = 0

                        if previous_position is not None:
                            contribution += (source_position - previous_position) / 2
                        if next_position is not None:
                            contribution += (next_position - source_position) / 2

                        #delta_xs = []

                        #if previous_position is not None:
                        #    delta_xs.append(source_position - previous_position)
                        #if next_position is not None:
                        #    delta_xs.append(next_position - source_position)

                        multiplicative_mask[0, 0, i, j] = contribution
                else:
                    raise ValueError('Invalid integration technique')

            #print(multiplicative_mask)

            # The transformer accepts an additive mask for the logit, so we compute the additive mask
            # as the log of the multiplicative mask, due to the fact that:
            # multiplicative_mask * exp(logit) = exp(logit + log(multiplicative_mask))
            mask = torch.log(multiplicative_mask + 1e-9)
        else:
            raise NotImplementedError('Seqlen must be greater than 1')

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    @torch.inference_mode()
    def forward_with_embeddings(self, tokens: torch.Tensor, start_pos: int, embedding_override = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens) if embedding_override is None else embedding_override
        standard_idx = torch.arange(start_pos, start_pos + seqlen, device=h.device)
        freqs_cis = compute_freqs_cis(standard_idx, self.params.dim // self.params.n_heads)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()