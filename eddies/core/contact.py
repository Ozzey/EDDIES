"""LOCAL linear compliant-contact analysis, not an identified Franka model."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def contact_matrix(stiffness,alpha,beta):
    if not np.isfinite([stiffness,alpha,beta]).all():
        raise ValueError('Require finite k, alpha, beta')
    if stiffness<=0 or alpha<=0 or not 0<beta<=1:
        raise ValueError('Require k>0, alpha>0, 0<beta<=1')
    return np.array([[1-beta,beta],[-alpha*stiffness,1.]])

def contact_certificate(stiffness,alpha,beta):
    A=contact_matrix(stiffness,alpha,beta)
    if not alpha*stiffness<1:
        raise ValueError('Jury sufficient-and-necessary condition alpha*k<1 fails')
    P=solve_discrete_lyapunov(A.T,np.eye(2))
    P=(P+P.T)/2                     # use (and certify) the symmetric matrix
    if np.linalg.eigvalsh(P).min()<=0: raise RuntimeError('Numerical Lyapunov solve failed')
    residual=A.T@P@A-P+np.eye(2)
    if np.linalg.norm(residual)>1e-9*(1+np.linalg.norm(A)**2*np.linalg.norm(P)):
        raise RuntimeError('Lyapunov residual too large')
    if np.linalg.eigvalsh(A.T@P@A-P).max()>=0:
        raise RuntimeError('A^T P A - P is not negative definite in floating point')
    S=np.linalg.cholesky(P).T
    contraction=float(np.linalg.norm(S@A@np.linalg.inv(S),2))
    if not contraction<1: raise RuntimeError('Contraction certificate not < 1 numerically')
    return {'A':A,'P':P,'contraction':contraction,
            'spectral_radius':float(np.max(np.abs(np.linalg.eigvals(A))))}
