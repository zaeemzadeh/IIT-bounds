from itertools import chain, combinations
import numpy as np
import pyphi.utils



def node_marginalizers(n_nodes, all_states=None):
    """Computes node marginalizers
    For each node, e.g. A, node marginalizer is a 2**n binary vector
    It is 1. when the state corresponding to it contibutes to pi(any single node|A=0)
    In other words, it is 1. whenever its corresponding node is 0.
    Args:
        all_states: all possible states of the system
    Returns:
        (np.ndarray[float]):  node marginalizers  nx2**n
    """
    if all_states is None:        
        all_states = list(pyphi.utils.all_states(n_nodes))
    marginalizers = 1 - np.array([list(state) for state in all_states])
    return marginalizers.transpose()


def order_k_marginalizer(k, node, node_marginalizers):
    """Computes marginalizers for all the mechanisms with size k that contain ''node''
    Each row corresponds to a mechanism M of size k that contain ''node''
    It is 1 when the state corresponding to it contibutes to pi(any signle node|M=0)
    In other words, it is 1 whenever all the nodes in M are 0.
    Args:
        k: mechanism size we are interested in
        node: individual node we are interested in, e.g. (0,).
        node_marginalizers: node marginalizers obtained using ''node_marginalizers''
    Returns:
        (np.ndarray[float]):  order k marginalizers 
    """
    n_nodes = len(node_marginalizers)
    ## finding all the mechanisms of size k that contain ''node''
    other_nodes = np.delete(np.arange(n_nodes), node)
    other_nodes_comb = list(combinations(other_nodes, k-1))
    marg = np.zeros(2**n_nodes)
    for comb in other_nodes_comb:
        comb_marg = node_marginalizers[comb + (node,),:]
        comb_marg = comb_marg.prod(axis=0)
        marg += comb_marg
    marg = marg > 0.
    return marg.astype(int)


def unique_order_marginalizers(node, node_marginalizers):
    """Finds the states that contribute to pi(any signle node|M=0) for |M| = k but not |M| > k. (M contains ''node'')
    
    Each row corresponds to a mechanism size k
    It is 1. when the state corresponding to it contibutes to pi(any signle node|M=0) for |M| = k but not |M| > k. (M contains ''node'')
    Args:
        node: individual node we are interested in, e.g. (0,).
        node_marginalizers: node marginalizers obtained using ''node_marginalizers''
    Returns:
        (np.ndarray[float]):  unique order marginalizers  nx2**n, largest order first
    """
    n_nodes = len(node_marginalizers)
    unique_order_marginalizers = []
    for order in range(n_nodes,0,-1):
        # finding all the states the contribute to pi(any signle node|M=0) for |M| = k
        order_k = np.array(order_k_marginalizer(order, node, node_marginalizers))
        
        # removing the ones that also occur in |M| > k
        if order < n_nodes:
            order_k -= np.minimum(np.array(unique_order_marginalizers).sum(axis=0),1)
        unique_order_marginalizers += [order_k]
    
    return np.array(unique_order_marginalizers)


def binomial_coeffs(n, k=None, normalize=False):
    """Computes all the binomial coefficients of n, i.e., n choose k for all k
    Uses multiplicative formula to avoid calculating factorials
    Args:
        n (int):  n in n choose k
        k (int):  set k to compute coefficients upto k (default: None, full sum)
        normalize (bool): if True divides the coefficient by 2**n (default: False)
    Returns:
        (np.ndarray[float128]):  binomial coefficients of n
    """
    coeffs = [np.float128(1.)]
    if normalize:
        coeffs[0] /= 2.**n
    
    if k is None:
        k = n
    for i in range(1, k+1):
        coeffs += [coeffs[-1]*(n+1-i)/i]

    return np.array(coeffs).astype(np.float128)

def sum_binomial_coeffs(n,k,normalize=False):
    """Computes partial sum of binomial coefficients of n    
    sum nCr(n,i) for i=0 ... k
    Args:
        n (int):  n in n choose k
        k (int):  upper limit of sum (inclusive)
        normalize (bool): if True divides the coefficient by 2**n (default: False)
    Returns:
        (float128):  partial sum of binomial coefficients of n up to k
    """
    return sum(binomial_coeffs(n,k, normalize))

def stirling_nCr(n, k):
    """Computes approximation of nCr using stirling approximation of factorial   
    Args:
        n (int):  n in n choose k
        k (int):  k in n choose k
    Returns:
        (float128):  approximation of nCr
    """
    approx = np.float128(1.)
    approx *= np.sqrt(n/(k*(n-k)))
    approx *= (n/(n-k))**n
    approx *= ((n-k)/k)**k
    approx /= np.sqrt(2*np.pi)
    return approx

def binary_entropy(p):
    """
    binary entropy with probability p  
    Args:
        p (float):  probability
    Returns:
        (float):  binary entropy with probability p
    """
    assert(p <=1. and p >= 0)
    if p == 1. or p == 0.:
        return 0.
    return -p*np.log2(p) -(1-p)*np.log2(1-p)