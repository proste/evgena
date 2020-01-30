import numpy as np


def nsgaII(fitnesses, pop_size):
    chosen = []
    for front in non_dominated_fronts(fitnesses):
        to_drop = max(0, (len(front) + len(chosen)) - pop_size)
        if not to_drop:
            chosen.extend(front)
        else:
            distances = crowding_distance(fitnesses[front, :])
            chosen.extend(front[distances.argsort()[to_drop:]])
    
    return chosen


def crowding_distance(fitnesses):
    """
    Parameters
    ----------
    fitnesses : np.ndarray
        array of shape (n_individuals, m_fitnesses)
    
    Returns
    -------
    np.ndarray
        array of shape (n_individuals,) of per individual crowding distances
    """
    front_size = len(fitnesses)
    if front_size <= 2:
        return np.array([np.inf] * front_size)
    
    distances = np.zeros(shape=(front_size,), dtype=np.float32)
    for fitness in fitnesses.T:
        order = fitness.argsort()
        sorted_fitness = fitness[order]
        
        span = sorted_fitness[-1] - sorted_fitness[0]
        distances[order] += np.r_[np.inf, ((sorted_fitness[2:] - sorted_fitness[:-2]) / span), np.inf]
    
    return distances


def non_dominated_fronts(fitnesses, method='fast'):
    """
    Parameters
    ----------
    fitnesses : np.ndarray
        array of shape (n_individuals, m_fitnesses),
        with rows sorted by (descending) significance
    method : {'fast'},  optional
        method to use for computation
        
    Returns
    -------
    Generator[np.ndarray]
        generator of fronts - arrays of indices of individuals
    """
    fnc = {
        'fast': _non_dominated_fronts_fast,
        
    }[method]
    
    yield from fnc(fitnesses)    
    

def _non_dominated_fronts_fast(fitnesses):
    # transpose for convenience
    fitnesses = fitnesses.T
    
    pop_size = fitnesses.shape[1]
    
    a_dominates_b = np.ones(shape=(pop_size, pop_size), dtype=np.bool)
    orderings = fitnesses.argsort(axis=1)
    for ordering, fitness in zip(orderings, fitnesses):
        sorted_fitness = fitness[ordering]
        for a_i, a in enumerate(ordering):
            # fitness strictly lower than others certainly not dominates them (beware of same valued fitness)
            a_dominates_b[a, ordering[(a_i + 1):]] &= (sorted_fitness[a_i] == sorted_fitness[(a_i + 1):])
    
    # same individuals do not dominate each other
    a_dominates_b[a_dominates_b & a_dominates_b.T] = False
    
    dominators = a_dominates_b.sum(axis=0)
    while True:
        mask = (dominators == 0)
        front = np.flatnonzero(mask)
        
        if front.size:
            dominators[mask] = -1
            dominators -= a_dominates_b[front, :].sum(axis=0)
            yield front
        else:
            break
