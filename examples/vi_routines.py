#Coupling Normalising flow with Bernstein Bijectors, based on distrax example.

import numpy as np
import jax
import jax.numpy as jnp

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
import haiku as hk
import distrax
from bernsteinbijectors import BernsteinBijector

Array = jnp.ndarray
PRNGKey = Array
OptState = Any


def make_conditioner(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int
) -> hk.Sequential:
  """Creates an."""

  #print('num bij prams', num_bijector_params)

  return hk.Sequential([
      hk.Flatten(preserve_dims=-len(event_shape)),
      hk.nets.MLP(hidden_sizes, activate_final=True),
      # We initialize this linear layer to zero so that the flow is initialized
      # to the identity function.
      hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros),
      hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
  ])

def make_flow_model(
    event_shape: Sequence[int],
    num_layers: int = 4,
    hidden_sizes: Sequence[int] = [250, 250],
    bernstein_degree: int = 20,
) -> distrax.Transformed:
    """Creates the flow model."""
    # Alternating binary mask.
    mask = np.arange(0, np.prod(event_shape)) % 2
    mask = np.reshape(mask, event_shape)
    mask = mask.astype(bool)

    def bijector_fn(params: Array):
        return BernsteinBijector(params)

    num_bijector_params = bernstein_degree

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape, hidden_sizes,
                                        num_bijector_params))
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = distrax.Inverse(distrax.Chain(layers))   #bijective transformation from latent space (uniform dist) to data space (target distribution)

    base_distribution = distrax.Independent(
        distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)*1),
        reinterpreted_batch_ndims=len(event_shape)
    )

    return distrax.Transformed(base_distribution, flow)
