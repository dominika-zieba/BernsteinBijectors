import distrax
import jax
import jax.numpy as jnp

from typing import Tuple
from distrax._src.bijectors import bijector as base

Array = base.Array

def bernstein_basis_polynomial(k, n):
    # returns a berstein basis polynomial b_{k,n}(x) = nCk x^k (1-x)^(n-k), nCk = n!/(k!(n-k)!), as a function
    # gammaln(n) = ln((n-1)!)
    binomial_coeff = jnp.exp(jax.scipy.special.gammaln(n+1) - (jax.scipy.special.gammaln(n-k+1)+jax.scipy.special.gammaln(k+1)))

    def basis_polynomial(x):
        return binomial_coeff * jnp.exp(k*jnp.log(x) + (n-k)*jnp.log(1-x))
        
    return basis_polynomial

def bernstein_basis_polynomial_derivative(k, n):
    def db_k_n_dx(x):  
        if k !=0:
            return n * (bernstein_basis_polynomial(k-1,n-1)(x) - bernstein_basis_polynomial(k,n-1)(x)) 
        else: 
            return -n*jnp.exp((n-1)*jnp.log(1-x))
    return db_k_n_dx


def bernstein_transform_fwd(x: Array, alphas: Array) -> Tuple[Array, Array]:
    #Bernstein transform of a scalar x
    
    clip = 1e-6
    x = jnp.clip(x, clip, 1.0 - clip)

    n = alphas.shape[-1] - 1 #degree of the bernstein polynomial
    bernstein_basis = [bernstein_basis_polynomial(k,n) for k in range(n+1)]

    basis_at_x = jnp.array([basis_polynomial(x) for basis_polynomial in bernstein_basis])
    y = jnp.sum(jnp.multiply(basis_at_x, alphas))

    bernstein_basis_derivatives = [bernstein_basis_polynomial_derivative(k, n) for k in range(n+1)]
    basis_derivative_at_x = jnp.atleast_2d(jnp.array([basis_derivative(x) for basis_derivative in bernstein_basis_derivatives])) #(poly_degree,)
    derivative_in_basis = jnp.sum(jnp.multiply(basis_derivative_at_x,alphas))
    logdet = jnp.log(jnp.abs(derivative_in_basis))

    return y, logdet

def bernstein_fwd(x:Array, alphas: Array) -> Array:

    #asssumes x is a scalar
    clip = 1e-6
    x = jnp.clip(x, clip, 1.0 - clip)
    n = alphas.shape[-1] - 1 #degree of the bernstein polynomial
    bernstein_basis = [bernstein_basis_polynomial(k,n) for k in range(n+1)]

    basis_at_x = jnp.array([basis_polynomial(x) for basis_polynomial in bernstein_basis])
    y = jnp.sum(jnp.multiply(basis_at_x, alphas))

    return y

def bernstein_log_det(x: Array, alphas: Array) -> Array:
    
    clip = 1e-6
    x = jnp.clip(x, clip, 1.0 - clip)

    n = alphas.shape[-1] - 1 #degree of the bernstein polynomial
    bernstein_basis_derivatives = [bernstein_basis_polynomial_derivative(k, n) for k in range(n+1)]
    basis_derivative_at_x = jnp.array([basis_derivative(x) for basis_derivative in bernstein_basis_derivatives]) #(poly_degree,)
    derivative_in_basis = jnp.sum(jnp.multiply(basis_derivative_at_x,alphas))

    logdet = jnp.log(jnp.abs(derivative_in_basis)) #log of the abosolute value of the derivative

    return logdet


def bernstein_transform_inv(y: Array, alphas: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y)."""
    clip = 1e-6
    y = jnp.clip(y, clip, 1.0 - clip)
    
    n_points = 200
    clip = 1e-6

    x_fit = jnp.linspace(clip, 1 - clip, n_points)
    y_fit = jax.vmap(bernstein_fwd, in_axes=(0, None))(x_fit, alphas)  #this is the y=f(x) values for the linear xs

    #interpolation function
    def inp(y, y_fit, x_fit):        
        return jnp.interp(y, y_fit, x_fit)

    x = inp(y, y_fit, x_fit)

    logdet = - bernstein_log_det(x,alphas)

    return x, logdet
    

def get_increasing_alphas_nd(unconstrained_params, range_min=0, range_max=1):
    #returns a strictly increasing sequence of alphas, alpha_0=range_min, alpha_n = range_max, n=len(unconstrained_params)
    #can try different methods
    unconstrained_params = jnp.atleast_2d(unconstrained_params)
    cumsum = jnp.cumsum(jnp.abs(unconstrained_params)+1e-10, axis = -1)
    scaled_params = range_min + cumsum/jnp.atleast_2d(cumsum[:,-1]).T * (range_max - range_min)
    alphas = jnp.concatenate([jnp.full((unconstrained_params.shape[0],1), range_min), scaled_params], axis = -1)
    return alphas


class BernsteinBijector(distrax.Bijector):
    """Initializes a Bernstein bijector."""

    def __init__(self, nn_params_out, range_min=0, range_max=1, event_ndims_in=0):
        super().__init__(event_ndims_in=event_ndims_in)
        #print('nn_params_out', nn_params_out.shape)
        #self.alphas = get_increasing_alphas_nd(nn_params_out, range_min, range_max)
        #print(self.alphas.val.shape)
        if len(nn_params_out.shape) == 2:
            self.alphas = get_increasing_alphas_nd(nn_params_out, range_min, range_max)

        #print(nn_params_out.shape)
        #if len(nn_params_out.shape) == 3:
        #    self.alphas = jax.vmap(get_increasing_alphas_nd, in_axes = (0, None, None))(nn_params_out, range_min, range_max)

    def forward(self, x):
        """Computes y = f(x)."""
        bernstein_polynomial = get_bernstein_polynomial_nd(self.alphas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)
        return bernstein_polynomial(x)

    def forward_log_det_jacobian(self, x):
        """Computes log|det J(f)(x)|."""
        bernstein_polynomial_jacobian_determinant = get_bernstein_polynomial_jacobian(self.alphas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)

        return bernstein_polynomial_jacobian_determinant(x)

    def inverse(self, y):
        """Computes x = f^{-1}(y)."""
        n_points = 200
        clip = 1e-7

        x_fit = jnp.linspace(clip, 1 - clip, n_points)[..., jnp.newaxis] * jnp.ones(
            (1,) + (y.shape[-1],))

        # this should be just 200 linearly spaces x's going between 0 and 1 (actually 1e-7 and )

        y_fit = self.forward(x_fit)  #this is the y=f(x) values for the linear xs

        #interpolation function
        def inp(y, y_fit, x_fit):        
            return jnp.interp(y, y_fit, x_fit)

        x = jax.vmap(inp, in_axes=(-1))(y, y_fit, x_fit)
        #x_inp = [jnp.interp(y[i], y_fit[:, i], x_fit[:,i]) for i in range(y_fit.shape[-1])]
        #x = jnp.stack(x_inp, axis=-1)
        #print('x after inverse', x.shape)

        return x
        #return jnp.reshape(x,y.shape)

    #def forward_and_log_det(self, x):
    #    """Computes y = f(x) and log|det J(f)(x)|."""
    #    y = self.forward(x)
    #    #print('x',x.shape)
    #    #print('y', y.shape)
    #    logdet = self.forward_log_det_jacobian(x)
    #    return y, logdet

    #def inverse_and_log_det(self, y):
    #    """Computes x = f^(-1)(y) and log|det J(f^(-1))(y)| = - log|det J(f))(x)|."""
    #    x = self.inverse(y)
    #    logdet = - self.forward_log_det_jacobian(x)
    #    #jax.debug.print('{y}', y=y.shape)
    #    return x, logdet
    

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        #print('x', x.shape)
        #print('alphas', self.alphas)
        fn = jnp.vectorize(bernstein_transform_fwd, excluded=frozenset(), signature='(),(n)->(),()')

        y, logdet = fn(x, self.alphas)
        #print('logdet',logdet.shape)
        #print('y', y.shape)
        #print('alphas', self.alphas)
        return y, logdet

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        fn = jnp.vectorize(bernstein_transform_inv, signature='(),(n)->(),()')
        x, logdet = fn(y, self.alphas)
        return x, logdet