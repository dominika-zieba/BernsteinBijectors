import jax
import jax.numpy as jnp


def bernstein_basis_polynomial(k, n):
    # returns a berstein basis polynomial b_{k,n}(x) = nCk x^k (1-x)^(n-k), nCk = n!/(k!(n-k)!), as a function
    # gamma(n) = (n-1)!
    #binomial_coeff = jax.scipy.special.gamma(n+1) / ((jax.scipy.special.gamma(n-k+1))*jax.scipy.special.gamma(k+1))
    jnp.exp(jax.scipy.special.gammaln(n+1) - (jax.scipy.special.gammaln(n-k+1)+jax.scipy.special.gammaln(k+1)))

    def basis_polynomial(x):
        #return binomial_coeff * x**k * (1-x)**(n-k)
        return  binomial_coeff * jnp.exp(k*jnp.log(x) + (n-k)*jnp.log(1-x))
        
    return basis_polynomial

def get_bernstein_polynomial(alphas):
    #reruns a bernstein polynomial B_n with parameters alphas = (alpha_0, alpha_1, ..., alpha_n) as a function of x
    #B_n(x) = sum_{k=0}^n alpha_k b_{k,n}, i.e. a linear combination of the basis bernstein polynomials b_{k,n} with coefficients 'alphas'
    n = alphas.shape[-1] - 1 #degree of the bernstein polynomial
    bernstein_basis = [bernstein_basis_polynomial(k,n) for k in range(n+1)]
    #print([k for k in range(n+1)])

    def B_n(x):
        basis_at_x = jnp.array([basis_polynomial(x) for basis_polynomial in bernstein_basis])
        return jnp.sum( jnp.multiply(basis_at_x, jnp.atleast_2d(alphas).T).T, axis=-1)
        
    return B_n

def get_bernstein_polynomial_nd(alphas):
    #reruns a bernstein polynomial B_n with parameters alphas = (alpha_0, alpha_1, ..., alpha_n) as a function of x
    #B_n(x) = sum_{k=0}^n alpha_k b_{k,n}, i.e. a linear combination of the basis bernstein polynomials b_{k,n} with coefficients 'alphas'
    n = alphas.shape[-1] - 1 #degree of the bernstein polynomial  #shape n_dims, poly_degree
    bernstein_basis = [bernstein_basis_polynomial(k,n) for k in range(n+1)]

    def B(x): #assumes x has shape (n_interp_points, n_dim), i.e. works with the forward method
        basis_at_x = jnp.atleast_3d(jnp.array([basis_polynomial(x) for basis_polynomial in bernstein_basis])) #poly_degree, n_samps, n_dims
        in_basis = jnp.einsum('ijk, ki -> ijk', basis_at_x, jnp.atleast_2d(alphas))
        return jnp.sum(in_basis, axis=0)
    
    return B

def get_increasing_alphas(unconstrained_params, range_min=0, range_max=1):
    #returns a strictly increasing sequence of alphas, alpha_0=range_min, alpha_n = range_max, n=len(unconstrained_params)
    #can try different methods
    scaled_params = range_min + jnp.cumsum(jnp.abs(unconstrained_params)+1e-4) / jnp.sum(jnp.abs(unconstrained_params)+1e-4) * (range_max - range_min) 
    alphas = jnp.concatenate([jnp.array([range_min]), scaled_params])
    #thetas: NN params, unconstrained real (not 0)..
    return alphas

def get_increasing_alphas_nd(unconstrained_params, range_min=0, range_max=1):
    #returns a strictly increasing sequence of alphas, alpha_0=range_min, alpha_n = range_max, n=len(unconstrained_params)
    #can try different methods
    unconstrained_params = jnp.atleast_2d(unconstrained_params)
    scaled_params = range_min + jnp.cumsum(jnp.abs(unconstrained_params)+1e-4, axis = -1) / jnp.atleast_2d(jnp.sum(jnp.abs(unconstrained_params)+1e-4, axis = -1)).T * (range_max - range_min) 
    alphas = jnp.concatenate([jnp.full((unconstrained_params.shape[0],1), range_min), scaled_params], axis = -1)
    #thetas: NN params, unconstrained real (not 0)..
    return alphas

def bernstein_basis_polynomial_derivative(k, n):
    #need to fix cases for k=0, n=0
    def db_k_n_dx(x):  #probably a problem for jax (i.e cases)
        if k !=0:
            return n * (bernstein_basis_polynomial(k-1,n-1)(x) - bernstein_basis_polynomial(k,n-1)(x)) 
        else: 
            return -n*(1-x)**(n-1)
    return db_k_n_dx

def get_bernstein_polynomial_jacobian(alphas):
    n = alphas.shape[-1] - 1 #degree of the bernstein polynomial  #shape n_dims, poly_degree
    bernstein_basis_derivatives = [bernstein_basis_polynomial_derivative(k, n) for k in range(n+1)]
    #print('alphas', alphas.shape)

    #def log_J_det(x): 
        #print('x', x.shape)
        #print('x.val', x.val.shape)
    #    basis_derivative_at_x = jnp.atleast_3d(jnp.array([basis_derivative(x) for basis_derivative in bernstein_basis_derivatives])) #poly_degree, n_samps, n_dims
        #print('basis_der_at_x', basis_derivative_at_x.shape)
        #print('basis_der_at_x.val', basis_derivative_at_x.val.shape)
    #    in_basis = jnp.einsum('ijk, ki -> ijk', basis_derivative_at_x, jnp.atleast_2d(alphas))
        #print('in_basis', in_basis.shape)
        #print('in_basis.val', in_basis.val.shape)
        #logdet = jnp.sum(jnp.log(jnp.abs((jnp.sum(in_basis, axis=0)))), axis=-1)
        #print(logdet.val, logdet.val.shape)

    #    return jnp.sum(jnp.log(jnp.abs((jnp.sum(in_basis, axis=0)))), axis=-1)
    
    def log_J_det(x): 
        #print('x', x.shape)
        #print('x.val', x.val.shape)
        basis_derivative_at_x = jnp.atleast_2d(jnp.array([basis_derivative(x) for basis_derivative in bernstein_basis_derivatives])) #poly_degree, n_samps, n_dims
        #print('basis_der_at_x', basis_derivative_at_x.shape)
        #print('basis_der_at_x.val', basis_derivative_at_x.val.shape)
        in_basis = jnp.einsum('ik, ki -> ik', basis_derivative_at_x, jnp.atleast_2d(alphas))
        #print('in_basis', in_basis.shape)
        #print('in_basis.val', in_basis.val.shape)
        #logdet = jnp.sum(jnp.log(jnp.abs((jnp.sum(in_basis, axis=0)))), axis=-1)
        #print(logdet.val, logdet.val.shape)

        return jnp.sum(jnp.log(jnp.abs((jnp.sum(in_basis, axis=0)))), axis=-1)
    
    return log_J_det