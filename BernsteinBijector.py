import distrax
import jax
import jax.numpy as jnp

from typing import Tuple

from distrax._src.bijectors import bijector as base

clip = 1e-6

Array = base.Array

def log_bernstein_basis_polynomial(k, n):
    # returns a berstein basis polynomial b_{k,n}(x) = nCk x^k (1-x)^(n-k), nCk = n!/(k!(n-k)!), as a function
    # gamma(n) = (n-1)!
    log_binomial_coeff = jax.scipy.special.gammaln(n + 1) - (jax.scipy.special.gammaln(n - k + 1) + jax.scipy.special.gammaln(k + 1))
    
    def log_basis_polynomial(x):
        return log_binomial_coeff + k * jnp.log1p(x-1) + (n - k) * jnp.log1p(-x) #log1p(x) = log(1+x), but more accurate for x close to 0.

    return log_basis_polynomial

def bernstein_basis_polynomial_derivative(k, n):
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
    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial

    #alphas[1:]  # alpha_1, alpha_2, ..., alpha_n
    #alphas[:-1] # alpha_0, alpha_1, ..., alpha_n-1

    log_bernstein_basis_at_x = jnp.stack([log_bernstein_basis_polynomial(k, n-1)(x) for k in range(n)])
    coeffs = n*(alphas[1:] - alphas[:-1])      #(alpha_k - alpha_k-1) for k between 0 and n

    logdet = jax.scipy.special.logsumexp(log_bernstein_basis_at_x, b=coeffs)
    return logdet

def log_abs_bernstein_basis_polynomial_derivative(k, n):
    #returns a function that computes the log_abs_derivative and sgn(derivative)
    def log_abs_db_k_n_dx(x):
        
        if k != 0: ##this might be negative in the log.. -> problem.
            #return jnp.log(n) + 
            #    log_bernstein_basis_polynomial(k - 1, n - 1)(x)
            #    + jnp.log1p(-jnp.exp(log_bernstein_basis_polynomial(k, n - 1)(x) - log_bernstein_basis_polynomial(k - 1, n - 1)(x))) 
            log_abs_der = jnp.log(n) + jnp.log(jnp.abs(jnp.exp(log_bernstein_basis_polynomial(k - 1, n - 1)(x)) 
                - jnp.exp(log_bernstein_basis_polynomial(k, n - 1)(x))))
            sgn_der = jnp.sign((jnp.exp(log_bernstein_basis_polynomial(k - 1, n - 1)(x)) - jnp.exp(log_bernstein_basis_polynomial(k, n - 1)(x))))
            
            return log_abs_der, sgn_der
        
        else:
            log_abs_der = jnp.log(n) + (n - 1) * jnp.log(1 - x)
            sgn_der = jnp.full(x.shape, -1)
            
            return log_abs_der, sgn_der

    return log_abs_db_k_n_dx



def get_increasing_alphas_nd(unconstrained_params, range_min=0, range_max=1):
    # returns a strictly increasing sequence of alphas, alpha_0=range_min, alpha_n = range_max, n=len(unconstrained_params)

    unconstrained_params = jnp.atleast_2d(unconstrained_params)
    cumsum = jnp.cumsum(jnp.abs(unconstrained_params) + 1e-5, axis=-1)
    scaled_params = range_min + cumsum / jnp.atleast_2d(cumsum[:, -1]).T * (
        range_max - range_min
    )
    alphas = jnp.concatenate(
        [jnp.full((unconstrained_params.shape[0], 1), range_min), scaled_params],
        axis=-1,
    )

    return alphas

def bernstein_log_transform_fwd(x: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|d/dx(f(x))|"""
    # takes in a scalar x

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    log_bernstein_basis = [log_bernstein_basis_polynomial(k, n) for k in range(n + 1)]

    log_basis_at_x = jnp.array(
        [log_basis_polynomial(x) for log_basis_polynomial in log_bernstein_basis]
    )
    
    y = jnp.exp(jax.scipy.special.logsumexp(log_basis_at_x, b=alphas)) #exp (log sum (alphas * exp(log_basis)))

    log_abs_bernstein_basis_derivatives = [
        log_abs_bernstein_basis_polynomial_derivative(k, n) for k in range(n + 1)
    ]
    log_abs_basis_derivative_at_x = jnp.atleast_2d(
        jnp.array(
            [log_abs_basis_derivative(x) for log_abs_basis_derivative in log_abs_bernstein_basis_derivatives]
        )
    )  # (poly_degree,)

    log_abs_derivative_in_basis =  jax.scipy.special.logsumexp(log_abs_basis_derivative_at_x[:,0], b=alphas*log_abs_basis_derivative_at_x[:,1])
   

    return y, log_abs_derivative_in_basis

def bernstein_transform_fwd(x: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|d/dx(f(x))|"""
    # takes in a scalar x

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    log_bernstein_basis = [log_bernstein_basis_polynomial(k, n) for k in range(n + 1)]

    log_basis_at_x = jnp.array(
        [log_basis_polynomial(x) for log_basis_polynomial in log_bernstein_basis]
    )
    
    y = jnp.exp(jax.scipy.special.logsumexp(log_basis_at_x, b=alphas)) #exp (log sum (alphas * exp(log_basis)))

    #bernstein_basis_derivatives = [
    #    bernstein_basis_polynomial_derivative(k, n) for k in range(n + 1)
    #]
    #basis_derivative_at_x = jnp.atleast_2d(
    #    jnp.array(
    #        [basis_derivative(x) for basis_derivative in bernstein_basis_derivatives]
    #    )
    #)  # (poly_degree,)

    #derivative_in_basis =  jnp.sum(jnp.multiply(basis_derivative_at_x, alphas))
    #logabsdet = jnp.log(jnp.abs(derivative_in_basis))
    logabsdet = bernstein_transform_log_derivative(x, alphas)

    return y, logabsdet

def bernstein_log_fwd(x: Array, alphas: Array) -> Array:
    """Computes y = f(x)."""
    # asssumes x is a scalar

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial
    log_bernstein_basis = [log_bernstein_basis_polynomial(k, n) for k in range(n + 1)]

    log_basis_at_x = jnp.array(
        [log_basis_polynomial(x) for log_basis_polynomial in log_bernstein_basis]
    )
    
    y = jnp.exp(jax.scipy.special.logsumexp(log_basis_at_x, b=alphas)) #exp (log sum (alphas * exp(log_basis)))

    return y

def bernstein_log_log_det(x: Array, alphas: Array) -> Array:

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial

    log_abs_bernstein_basis_derivatives = [
        log_abs_bernstein_basis_polynomial_derivative(k, n) for k in range(n + 1)
    ]

    #derivatives and signs..
    log_abs_basis_derivative_at_x = jnp.atleast_2d(
        jnp.array(
            [log_abs_basis_derivative(x) for log_abs_basis_derivative in log_abs_bernstein_basis_derivatives]
        )
    )  # (poly_degree,)

    log_abs_derivative_in_basis =  jax.scipy.special.logsumexp(log_abs_basis_derivative_at_x[:,0], b=alphas*log_abs_basis_derivative_at_x[:,1])

    return log_abs_derivative_in_basis

def bernstein_log_det(x: Array, alphas: Array) -> Array:

    n = alphas.shape[-1] - 1  # degree of the bernstein polynomial

    bernstein_basis_derivatives = [
        bernstein_basis_polynomial_derivative(k, n) for k in range(n + 1)
    ]
    basis_derivative_at_x = jnp.atleast_2d(
        jnp.array(
            [basis_derivative(x) for basis_derivative in bernstein_basis_derivatives]
        )
    ) 
    derivative_in_basis =  jnp.sum(jnp.multiply(basis_derivative_at_x, alphas))
    logabsdet = jnp.log(jnp.abs(derivative_in_basis))

    return logabsdet

def bernstein_log_transform_inv(y: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and  log|d/dy(f^-1(y))|."""
    n_points = 200
    clip = 1e-7

    x_fit = jnp.linspace(clip, 1 - clip, n_points)
    y_fit = jax.vmap(bernstein_log_fwd, in_axes=(0, None))(
        x_fit, alphas
    )  # this is the y=f(x) values for the linear xs

    # interpolation function
    def inp(y, y_fit, x_fit):
        return jnp.interp(y, y_fit, x_fit)

    x = inp(y, y_fit, x_fit)

    logdet = -bernstein_log_log_det(x, alphas)

    return x, logdet

def bernstein_transform_inv(y: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and  log|d/dy(f^-1(y))|."""
    n_points = 200
    clip = 1e-7

    x_fit = jnp.linspace(clip, 1 - clip, n_points)
    y_fit = jax.vmap(bernstein_log_fwd, in_axes=(0, None))(
        x_fit, alphas
    )  # this is the y=f(x) values for the linear xs

    # interpolation function
    def inp(y, y_fit, x_fit):
        return jnp.interp(y, y_fit, x_fit)

    x = inp(y, y_fit, x_fit)

    logdet = -bernstein_log_det(x, alphas)

    return x, logdet

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
        #x = jnp.clip(x, clip, 1.0 - clip)
        fn = jnp.vectorize(
            bernstein_transform_fwd, excluded=frozenset(), signature="(),(n)->(),()"
        )
        y, logdet = fn(x, self.alphas)

        return y, logdet

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        #y = jnp.clip(y, clip, 1.0 - clip)
        fn = jnp.vectorize(bernstein_transform_inv, signature="(),(n)->(),()")
        x, logdet = fn(y, self.alphas)
        return x, logdet