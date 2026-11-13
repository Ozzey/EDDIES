"""Conservative proposition alphabets from trustworthy predicate intervals."""
from itertools import product
import numpy as np

def possible_letters(bounds,max_unknown=12):
    """Predicate p is true iff h_p >= 0. Unknown intervals branch, never guess.

    Correlations are intentionally ignored: this is an overapproximation.
    The caller must establish interval coverage, including sensor uncertainty.
    """
    definite=set(); unknown=[]
    for name,(lower,upper) in bounds.items():
        if not np.isfinite([lower,upper]).all() or lower>upper: raise ValueError('Invalid predicate interval')
        if lower>=0: definite.add(name)
        elif upper>=0: unknown.append(name)
    if len(unknown)>max_unknown: raise RuntimeError('Uncertain-label branch cap exceeded; refine sensing/cells')
    return tuple(frozenset(definite|{name for name,b in zip(unknown,bits) if b})
                 for bits in product([False,True],repeat=len(unknown)))
