# Copyright 2020 The FlaxBERT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention Layers optimized for efficiency.

This file continues a journey of optimized attention implementations that
started in the trax framework; see
https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py

Implementation notes:
1. Many attention implementations compute O(n^2) query-key dot products all in
   parallel, which can easily use up all available memory. However, there is no
   requirement to compute all dot products in parallel, and instead attention
   can be run for a subset of queries at a time. The attention implementations
   here are designed to have configurable chunking. Further optimizatons such
   as local attention and LSH attention are primarily aimed at reducing training
   time, and not memory usage.
2. Once chunking is in place, the next potential way to run out of memory is to
   simultaneously instantiate queries, keys, and values for all heads at the
   same time. Transformers are typically tuned such that
   num_heads * d_attention_key == d_model. Since attention involves queries,
   keys, and values, the memory to store them can be ~3x the memory needed to
   store the input activations. Therefore, each chunk of the computation is
   responsible for its own query/key/value/output projections.
3. Attention masking is implemented by associating an integer (typically, the
   sequence position) with each query and key vector, and defining a function
   to compute attention masks from this information. The flax attention
   built-ins pass around O(n^2)-size attention mask tensors instead, which is
   not scalable for long sequences. Many Transformer implementations opt to
   compute this large mask tensor once and then re-use it across all layers of
   the model. This can save on compute, but it incurs a memory cost that also
   impacts the maximum memory available to other layers (e.g. feed-forward and
   output softmax layers). Computing full masks on-demand may be a bit slower,
   but we deem this tradeoff worth it because of the memory savings it brings.
4. It is our observation that for long sequences, the speed of an attention
   mechanism is limited not by the number of floating point operations (such as
   dot products), but rather by memory access speeds.
"""

import functools

from flax import nn
import multihead
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


NEG_INFINITY = -1e9


class MultiHeadWrapper(nn.Module):
  """Wrapper for batching attention across examples and heads."""

  def apply(self, *args, wrapped_module,
            num_heads=1, num_parallel_heads=None, use_python_loop=False,
            **kwargs):
    # Re-use the same rng key across all examples and heads. This will result in
    # broadcasted dropout, which saves memory.
    # TODO(kitaev): options to swap broadcasted RNG on/off
    rng = nn.make_rng() if nn.is_stochastic() else None

    def init_single_head(init_rng, args, kwargs):
      if rng is None:
        _, head_params = wrapped_module.init(init_rng, *args, **kwargs)
      else:
        with nn.stochastic(rng):
          _, head_params = wrapped_module.init(init_rng, *args, **kwargs)
      return head_params

    def init_wrapped_module(rng, unused_shape):
      single_example_args = jax.tree_map(lambda x: x[:1], args)
      return multihead.chunked_multihead_map(
          init_single_head,
          in_has_batch_dim=(False, True, False),
          in_has_head_dim=(True, False, False),
          out_has_batch_dim=False,
          out_has_head_dim=True,
          use_python_loop=True,
          )(jax.random.split(rng, num_heads), single_example_args, kwargs)
    # TODO(kitaev): The original intent was to have this be a transparent module
    # but for some reason naming this parameter '0' and inheriting from
    # nn.base.TransparentModule is not enough to stop this parameter name from
    # explicitly showing up in the parameter tree.
    params = self.param('attn', None, init_wrapped_module)

    def run_single_example_and_head(params, args, kwargs):
      if rng is None:
        return wrapped_module.call(params, *args, **kwargs)
      else:
        with nn.stochastic(rng):
          return wrapped_module.call(params, *args, **kwargs)

    return multihead.chunked_multihead_map(
        run_single_example_and_head,
        in_has_batch_dim=(False, True, False),
        in_has_head_dim=(True, False, False),
        out_has_batch_dim=True,
        out_has_head_dim=False,
        num_parallel_heads=num_parallel_heads,
        use_python_loop=use_python_loop,
    )(params, args, kwargs)


def make_multihead(module_type):
  return MultiHeadWrapper.partial(wrapped_module=module_type)


class ManuallyBatchedAttentionWrapper(nn.Module):
  """Wrapper for manually batched attention."""

  def apply(self, *args, wrapped_module, **kwargs):
    # An extra 'attn' scope is needed to match param structure with attention
    # types that use make_multihead.
    return wrapped_module(*args, name='attn', **kwargs)


def not_multihead(module_type):
  return ManuallyBatchedAttentionWrapper.partial(wrapped_module=module_type)


@make_multihead
class BertSelfAttention(nn.Module):
  """Masked dot-product self-attention."""

  def apply(self,
            hidden_states, mask=None, *,
            d_qkv=64,
            attention_dropout_rate=0.0,
            output_dropout_rate=0.0,
            deterministic=False,
            kernel_init=nn.linear.default_kernel_init,
            output_kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            bias=True):
    """Applies attention for a single batch element and head."""
    d_model = hidden_states.shape[-1]
    dense = nn.DenseGeneral.partial(
        axis=-1,
        features=(d_qkv,),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias)
    query, key, value = (dense(hidden_states, name='query'),
                         dense(hidden_states, name='key'),
                         dense(hidden_states, name='value'))
    attention_scores = jnp.einsum('TN,FN->FT', key, query)
    attention_scores = attention_scores / jnp.sqrt(d_qkv)
    if mask is not None:
      padding_mask = (1.0 - mask[None, :]) * NEG_INFINITY
      attention_scores = attention_scores + padding_mask
    attention_scores = nn.softmax(attention_scores)
    attention_probs = nn.dropout(
        attention_scores, rate=attention_dropout_rate,
        deterministic=deterministic)
    hidden_states = jnp.einsum('FT,TH->FH', attention_probs, value)
    hidden_states = nn.linear.DenseGeneral(
        hidden_states,
        features=d_model,
        axis=(-1,),
        kernel_init=output_kernel_init,
        name='output')
    hidden_states = nn.dropout(
        hidden_states, rate=output_dropout_rate, deterministic=deterministic)
    return hidden_states

