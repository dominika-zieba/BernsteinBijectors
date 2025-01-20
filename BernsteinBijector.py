from bernstein_routines import *

import distrax
import jax
import jax.numpy as jnp

class BernsteinBijector(distrax.Bijector):
    """Initializes a Bernstein bijector."""

    def __init__(self, nn_params_out, range_min=0, range_max=1):
        super().__init__(event_ndims_in=0)
        #print('nn_params_out', nn_params_out.shape)
        self.alphas = get_increasing_alphas_nd(nn_params_out, range_min, range_max)
        #print(self.alphas.val.shape)

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

    def forward_and_log_det(self, x):
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.forward(x)
        logdet = self.forward_log_det_jacobian(x)
        return y, logdet

    def inverse_and_log_det(self, y):
        """Computes x = f^(-1)(y) and log|det J(f^(-1))(y)| = - log|det J(f))(x)|."""
        x = self.inverse(y)
        logdet = - self.forward_log_det_jacobian(x)
        #jax.debug.print('{y}', y=y.shape)
        return x, logdet