from __future__ import annotations
import math


def wilson(successes:int,n:int,z:float=1.959963984540054):
    if (int(successes)!=successes or int(n)!=n or n<=0 or not 0<=successes<=n
            or not math.isfinite(z) or z<=0): raise ValueError("Invalid binomial counts or z")
    p=successes/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,c-h),min(1.,c+h)


def zero_failure_upper(n:int,alpha=.05):
    """One-sided exact iid Bernoulli upper bound, NOT a safety proof."""
    if n<=0 or not 0<alpha<1: raise ValueError("Invalid confidence input")
    return 1-alpha**(1/n)
