import distrax
import jax
import jax.numpy as jnp

from typing import Tuple

from distrax._src.bijectors import bijector as base

clip = 0.

Array = base.Array

def log_bernstein_basis_polynomial(k, n):
    """Returns a berstein basis polynomial b_{k,n}(x) = nCk x^k (1-x)^(n-k), nCk = n!/(k!(n-k)!), as a function of x"""
    # gamma(n) = (n-1)!
    log_binomial_coeff = jax.scipy.special.gammaln(n + 1) - (jax.scipy.special.gammaln(n - k + 1) + jax.scipy.special.gammaln(k + 1))
    
    #log1p(x) = log(1+x), but more accurate for x close to 0.
    def log_basis_polynomial(x):
        if k == 0:
            return jnp.where(
                x == 0, 0.,
                log_binomial_coeff + k * jnp.log1p(x-1) + (n - k) * jnp.log1p(-x),
            )
        if k == n:
            return jnp.where(
                x == 1, 0.,
                log_binomial_coeff + k * jnp.log1p(x-1) + (n - k) * jnp.log1p(-x),
            )
        else:
            return log_binomial_coeff + k * jnp.log1p(x-1) + (n - k) * jnp.log1p(-x)

    return log_basis_polynomial

def bernstein_basis_polynomial_derivative(k, n):
    """Returns a berstein basis polynomial derivative b'_{k,n}(x), as a function of x"""
    def db_k_n_dx(x):
        if k != 0:
            return n * (
                jnp.exp(log_bernstein_basis_polynomial(k - 1, n - 1)(x))
                - jnp.exp(log_bernstein_basis_polynomial(k, n - 1)(x))
            )
        else:
            return -n * jnp.exp((n - 1) * jnp.log(1 - x))

    return db_k_n_dx

def bernstein_transform_log_derivative(x: Array, alphas: Array) -> Array:
    """Returns a logaritm of the basis polynomial derivative, log(b'_{k,n})(x) at x"""
    # expects x to be a scalar

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial

    #alphas[1:]  # alpha_1, alpha_2, ..., alpha_n
    #alphas[:-1] # alpha_0, alpha_1, ..., alpha_n-1

    log_bernstein_basis_at_x = jnp.stack([log_bernstein_basis_polynomial(k, n-1)(x) for k in range(n)])
    coeffs = jnp.abs(n*(alphas[1:] - alphas[:-1]))      #(alpha_k - alpha_k-1) for k between 0 and n

    logdet = jax.scipy.special.logsumexp(log_bernstein_basis_at_x, b=coeffs)

    return logdet

def get_increasing_alphas_nd(unconstrained_params, range_min=0, range_max=1):
    """Returns a strictly increasing sequence of alphas, alpha_0=range_min, alpha_n = range_max, n=len(unconstrained_params)"""

    unconstrained_params = jnp.atleast_2d(unconstrained_params)
    cumsum = jnp.cumsum(jnp.abs(unconstrained_params) + 1e-11, axis=-1)
    scaled_params = range_min + cumsum / jnp.atleast_2d(cumsum[:, -1]).T * (
        range_max - range_min
    )
    alphas = jnp.concatenate(
        [jnp.full((unconstrained_params.shape[0], 1), range_min), scaled_params],
        axis=-1,
    )

    return alphas

def bernstein_transform_fwd(x: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|d/dx(f(x))|"""
    # expects x to be a scalar

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    log_basis_at_x = jnp.stack([log_bernstein_basis_polynomial(k, n)(x) for k in range(n + 1)])

    y = jnp.exp(jax.scipy.special.logsumexp(log_basis_at_x, b=alphas)) #exp (log sum (alphas * exp(log_basis)))

    logabsdet = bernstein_transform_log_derivative(x, alphas)

    return y, logabsdet

def bernstein_fwd(x: Array, alphas: Array) -> Array:
    """Computes y = f(x)."""
    # asssumes x is a scalar

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    log_basis_at_x = jnp.stack([log_bernstein_basis_polynomial(k, n)(x) for k in range(n + 1)])
    
    y = jnp.exp(jax.scipy.special.logsumexp(log_basis_at_x, b=alphas)) #exp (log sum (alphas * exp(log_basis)))

    return y

def bernstein_transform_inv(y: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and  log|d/dy(f^-1(y))|."""
    n_points = 200
    clip = 1e-7 ##TODO: get rid of this (currently runs into numerical issues if not clipped)

    x_fit = jnp.linspace(clip, 1 - clip, n_points)
    y_fit = jax.vmap(bernstein_fwd, in_axes=(0, None))(
        x_fit, alphas
    )  # this is the y=f(x) values for the linear xs

    # interpolation function
    def inp(y, y_fit, x_fit):
        return jnp.interp(y, y_fit, x_fit)

    x = inp(y, y_fit, x_fit)

    logdet = -bernstein_transform_log_derivative(x, alphas)

    return x, logdet

#Optimised Inverse Routines (root finder instead of linear interpolation (before))
import optimistix as optx

def inverse_cost(x: Array, args: Tuple[Array, Array]) -> Array:
    """Computes cost = y - f(x)"""
    y, alphas = args
    return y - bernstein_fwd(x,alphas)

def bernstein_optim_inv(y: Array, alphas: Array) -> Array:
    """Computes x = f^{-1}(y)"""
    #solver = optx.Newton(rtol=1e-7, atol=1e-6)
    #sol = optx.root_find(inverse_cost, solver, y0 = y, args=(y, alphas))
    solver = optx.Bisection(rtol=1e-6, atol=1e-6)
    sol = optx.root_find(inverse_cost, solver, y0 = y, args=(y, alphas), options = dict(lower=0, upper=1),  max_steps = 256, throw=False)
    return sol.value

def bernstein_optim_transform_inv(y: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and  log|d/dy(f^-1(y))|."""
    x = bernstein_optim_inv(y, alphas)
    logabsdet = - bernstein_transform_log_derivative(x, alphas) 
    return x, logabsdet

#Distrax Bijector class

class BernsteinBijector(distrax.Bijector):
    """Initializes a Bernstein bijector."""

    def __init__(self, nn_params_out, range_min=0, range_max=1, event_ndims_in=0):
        super().__init__(event_ndims_in=event_ndims_in)

        if len(nn_params_out.shape) == 2:
            self.alphas = get_increasing_alphas_nd(nn_params_out, range_min, range_max)

        if len(nn_params_out.shape) == 3:
            self.alphas = jax.vmap(get_increasing_alphas_nd, in_axes=(0, None, None))(
                nn_params_out, range_min, range_max
            )

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        fn = jnp.vectorize(
            bernstein_transform_fwd, excluded=frozenset(), signature="(),(n)->(),()"
        )
        y, logdet = fn(x, self.alphas)

        return y, logdet
    
    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        fn = jnp.vectorize(bernstein_optim_transform_inv, signature="(),(n)->(),()")
        x, logdet = fn(y, self.alphas)
        return x, logdet