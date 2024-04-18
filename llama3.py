# Llama 3 remake for educational purposes to help understand and experiment with the Llama3 model

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    """
    Applies RMS normalization to the input tensor.
    
    Args:
    dim (int): The feature dimension of the input tensor.
    eps (float, optional): A small value added to the denominator for numerical stability, default is 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Normalize the input tensor using RMS normalization and scale by learned weights.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The normalized and scaled tensor.
        """
        # Calculate the root mean square of the tensor along the last dimension
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize the input tensor and scale it by the learned weights
        normalized = x / rms
        return normalized * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Compute the complex sinusoidal frequencies for rotary position embeddings.
    This involves creating a matrix of frequencies that combines time and position dimensions
    using complex numbers to represent their interaction.
    
    Args:
    dim (int): Dimension size, usually the model's dimension or a multiple of it.
    end (int): The number of timesteps or positions for which to compute the frequencies.
    theta (float, optional): The base of the exponent for computing frequency values, defaults to 10000.0.
    
    Returns:
    torch.Tensor: A tensor of complex numbers representing the rotary frequencies.
    """
    # Generate the first part of the frequency calculation using a power-law decay based on theta.
    indices = torch.arange(0, dim, 2).float()
    freqs = 1.0 / torch.pow(theta, indices / dim)

    # Generate the time steps
    timesteps = torch.arange(end, dtype=torch.float32)

    # Calculate outer product to combine frequencies and timesteps
    freq_matrix = torch.outer(timesteps, freqs)

    # Convert the frequency matrix to complex numbers where real parts are ones and imaginary parts are the frequency values
    freqs_cis = torch.polar(torch.ones_like(freq_matrix), freq_matrix)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape freqs_cis for broadcasting to match the dimensions of tensor x,
    except for the 2nd and last dimensions which align with freqs_cis's first and second dimensions respectively.
    """
    # Ensure the input dimensions and shape conditions are met
    assert x.ndim >= 2, "Tensor x must have at least two dimensions"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "Shape mismatch for broadcasting"

    # Define the target shape for freqs_cis to broadcast correctly
    target_shape = [1] * x.ndim
    target_shape[1] = x.shape[1]   # Match dimension 2 (0-based index)
    target_shape[-1] = x.shape[-1] # Match the last dimension

    return freqs_cis.view(target_shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Process xq and xk to get complex tensors from their real counterparts
    def to_complex(x):
        return torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))

    # Apply rotary embeddings using frequency coefficients
    def apply_embedding(x_complex, freqs_cis):
        freqs_cis_adjusted = reshape_for_broadcast(freqs_cis, x_complex)
        return torch.view_as_real(x_complex * freqs_cis_adjusted).flatten(start_dim=3)

    xq_complex = to_complex(xq)
    xk_complex = to_complex(xk)
    xq_embedded = apply_embedding(xq_complex, freqs_cis)
    xk_embedded = apply_embedding(xk_complex, freqs_cis)

    # Ensure the output tensors match the input type
    return xq_embedded.type_as(xq), xk_embedded.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat elements of a tensor along the third dimension by n_rep times.
    """
    if n_rep == 1:
        return x
    # Directly use torch.repeat_interleave to repeat the elements along the specified dimension
    return torch.repeat_interleave(x, repeats=n_rep, dim=2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # Initialize caches; consider potential adjustments to dimensions if necessary
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim),
            device='cuda'
        )
        self.cache_v = torch.zeros_like(self.cache_k)

    def forward(self, x, start_pos, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Ensure proper indexing during cache updates
        self.update_caches(xk, xv, bsz, start_pos, seqlen)

        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        scores = F.softmax(scores, dim=-1).type_as(xq)

        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def update_caches(self, xk, xv, bsz, start_pos, seqlen):
        # Correctly assign the slices for batch size to match xk and xv's actual batch size
        self.cache_k[:bsz, start_pos:start_pos+seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos+seqlen] = xv


class FeedForward(nn.Module):
    """
    A feedforward neural network module with parameterizable dimensions and a nonlinear activation function.
    This module uses custom dimensions for hidden layers, adjusted to be multiples of a specified base.
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        # Calculate the adjusted hidden dimension size
        adjusted_hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            adjusted_hidden_dim = int(ffn_dim_multiplier * adjusted_hidden_dim)
        # Ensure the hidden dimension is a multiple of `multiple_of`
        adjusted_hidden_dim = (adjusted_hidden_dim + multiple_of - 1) // multiple_of * multiple_of

        # Initialize weights for the three linear transformations
        self.init_linear_layers(dim, adjusted_hidden_dim)

    def init_linear_layers(self, dim: int, hidden_dim: int):
        """
        Initializes the linear layers with specified input and hidden dimensions.
        """
        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False)

    def forward(self, x):
        """
        Forward pass of the feedforward network applying a nonlinear activation function (SiLU)
        between two linear transformations and combining results from an element-wise multiplication.
        """
        # Apply the first transformation and activation
        x_transformed = F.silu(self.w1(x))
        # Apply a second transformation for modulation
        x_modulated = self.w3(x)
        # Element-wise multiplication of the transformed and modulated outputs
        x_combined = x_transformed * x_modulated
        # Final transformation and return the result
        return self.w2(x_combined)


class TransformerBlock(nn.Module):
    """
    A single Transformer block that includes a normalization and attention mechanism,
    followed by a feed-forward network, typically used in transformer architectures.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.initialize_layers(args)

    def initialize_layers(self, args):
        """
        Initialize components of the Transformer block including the attention mechanism,
        normalization layers, and the feed-forward network.
        """
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=4 * self.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier
        )
        
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask=None):
        """
        Forward pass through the Transformer block applying attention and feed-forward network.

        Args:
        x (torch.Tensor): Input tensor to the Transformer block.
        start_pos (int): Start position index for positional encoding.
        freqs_cis (torch.Tensor): Frequency coefficients for rotary positional embeddings.
        mask (Optional[torch.Tensor]): Mask for the attention operation to ignore certain positions.

        Returns:
        torch.Tensor: Output tensor after passing through the Transformer block.
        """
        # Apply attention normalization and attention mechanism
        normalized_attention = self.attention_norm(x)
        attention_output = self.attention(normalized_attention, start_pos, freqs_cis, mask)

        # Sum input with attention output (residual connection)
        residual_connection = x + attention_output

        # Apply feed-forward normalization and feed-forward network
        normalized_ff = self.ffn_norm(residual_connection)
        ff_output = self.feed_forward(normalized_ff)

        # Sum the residual connection output with feed-forward output (second residual connection)
        return residual_connection + ff_output


class Transformer(nn.Module):
    """
    A Transformer model that includes an embedding layer, multiple Transformer blocks,
    and a final normalization and output layer for sequence processing.
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.initialize_embeddings()
        self.initialize_transformer_blocks()
        self.initialize_output_layers()

    def initialize_embeddings(self):
        """Initialize the token embedding layer."""
        self.tok_embeddings = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim, init_method=lambda x: x
        )

    def initialize_transformer_blocks(self):
        """Initialize the Transformer blocks."""
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(i, self.params) for i in range(self.params.n_layers)]
        )
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
            self.params.rope_theta
        )

    def initialize_output_layers(self):
        """Initialize the final output layer."""
        self.output = ColumnParallelLinear(
            self.params.dim, self.params.vocab_size, bias=False, init_method=lambda x: x
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Forward pass through the Transformer model using provided tokens and starting position.

        Args:
        tokens (torch.Tensor): Input token IDs.
        start_pos (int): Starting position for slicing the precomputed frequency embeddings.

        Returns:
        torch.Tensor: The output logits from the Transformer.
        """
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + tokens.size(1)].to(tokens.device)
        mask = self.create_attention_mask(tokens.size(1), start_pos, tokens.device)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        
        h = self.norm(h)
        return self.output(h).float()

    def create_attention_mask(self, sequence_length, start_pos, device):
        """Create an attention mask to ignore future tokens in self-attention layers."""
        if sequence_length > 1:
            mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((sequence_length, start_pos), device=device), mask])
            return mask.type_as(mask)
        return None
