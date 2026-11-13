"""Online learning separated from safety: nominal RLS and nested feasible models.

Noise bounds and the initial model enclosure must be supplied from outside this
learner. Residual maxima are not promoted to worst-case certificates.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from .certificates import vector


class ModelInconsistency(RuntimeError):
    pass


LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


class SetMembershipModel:
    """Nested polytopic parameter sets (Theorem 4).  Intervals are floating-point
    LP estimates (HiGHS, tolerance 1e-10), not interval-arithmetic enclosures.
    Emptiness is decided by a phase-1 LP with an explicit, reported slack."""
    def __init__(self, lower, upper, epsilon, empty_tolerance=1e-9):
        self.lower=np.asarray(lower,float); self.upper=np.asarray(upper,float)
        if (self.lower.ndim!=2 or self.upper.shape!=self.lower.shape or
                not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all() or
                np.any(self.lower>self.upper)):
            raise ValueError("Expected finite coefficient bounds of shape [output, features]")
        self.empty_tolerance=float(empty_tolerance)
        self.last_slack=0.0
        self.epsilon=np.broadcast_to(np.asarray(epsilon,float),(self.lower.shape[0],)).copy()
        if np.any(self.epsilon<0) or not np.isfinite(self.epsilon).all(): raise ValueError("Invalid disturbance bound")
        self.phi=[]; self.y=[]
        self.version=0

    def _matrices(self, out, phi=None,y=None):
        features=self.phi if phi is None else phi; outputs=self.y if y is None else y
        if not features: return None,None
        f=np.asarray(features); values=np.asarray(outputs)[:,out]; eps=self.epsilon[out]
        return np.concatenate([f,-f]),np.concatenate([values+eps,-values+eps])

    def _solve(self,objective,out,phi=None,y=None):
        a,b=self._matrices(out,phi,y)
        bounds=list(zip(self.lower[out],self.upper[out]))
        r=linprog(objective,A_ub=a,b_ub=b,bounds=bounds,method='highs',options=LP_OPTIONS)
        if not r.success:   # HiGHS presolve can misjudge near-degenerate LPs: retry without it
            r=linprog(objective,A_ub=a,b_ub=b,bounds=bounds,method='highs',
                      options={**LP_OPTIONS,"presolve":False})
        return r

    def _phase1_slack(self,out,phi,y):
        """min t s.t. A theta - b <= t, lower <= theta <= upper (t* <= 0 iff nonempty)."""
        a,b=self._matrices(out,phi,y)
        n=a.shape[1]
        A=np.hstack([a,-np.ones((len(a),1))])
        bounds=list(zip(self.lower[out],self.upper[out]))+[(None,None)]
        c=np.r_[np.zeros(n),1.0]
        r=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs',options=LP_OPTIONS)
        if not r.success:
            r=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs',options={**LP_OPTIONS,"presolve":False})
        if not r.success: raise ModelInconsistency("Phase-1 LP failed: "+str(r.message))
        return float(r.fun)

    def update(self, phi, y):
        phi,y=vector(phi),vector(y)
        if len(phi)!=self.lower.shape[1] or len(y)!=self.lower.shape[0]: raise ValueError("Dimension mismatch")
        new_phi=self.phi+[phi.copy()]; new_y=self.y+[y.copy()]
        # Transactional: a contradictory sample NEVER silently replaces the model.
        slack=0.0
        for out in range(len(y)):
            t=self._phase1_slack(out,new_phi,new_y)
            scale=1.0+float(np.max(np.abs(np.asarray(new_y)[:,out])))
            if t>self.empty_tolerance*scale:
                raise ModelInconsistency(f"Empty model set (phase-1 slack {t:.3e}): stop, diagnose and recertify")
            slack=max(slack,t)
        self.phi,self.y=new_phi,new_y; self.version+=1; self.last_slack=slack

    def predict_interval(self,phi,include_noise=True):
        phi=vector(phi)
        if len(phi)!=self.lower.shape[1]: raise ValueError("Dimension mismatch")
        lo=[]; hi=[]
        for out in range(self.lower.shape[0]):
            mn=self._solve(phi,out); mx=self._solve(-phi,out)
            if not mn.success or not mx.success: raise ModelInconsistency("LP failed")
            eps=self.epsilon[out] if include_noise else 0
            l,h=mn.fun-eps,-mx.fun+eps
            if l>h+1e-9*(1+abs(h)): raise ModelInconsistency("Inverted prediction interval")
            lo.append(min(l,h)); hi.append(max(l,h))
        return np.asarray(lo),np.asarray(hi)


class RecursiveLeastSquares:
    """Nominal predictor only; estimates do not certify safety."""
    def __init__(self,n_features,n_outputs,ridge=1.0,forgetting=1.0,max_trace=1e8):
        if (not np.isfinite(ridge) or ridge<=0 or not 0<forgetting<=1
                or not np.isfinite(max_trace) or max_trace<=0):
            raise ValueError("Invalid RLS parameters")
        self.theta=np.zeros((n_outputs,n_features)); self.P=np.eye(n_features)/ridge
        self.forgetting=forgetting; self.max_trace=float(max_trace)
    def predict(self,phi):
        phi=vector(phi)
        if len(phi)!=self.theta.shape[1]: raise ValueError("Dimension mismatch")
        return self.theta@phi
    def update(self,phi,y):
        phi,y=vector(phi),vector(y)
        if len(phi)!=self.theta.shape[1] or len(y)!=self.theta.shape[0]:
            raise ValueError("Dimension mismatch")
        gain=self.P@phi/(self.forgetting+phi@self.P@phi)
        self.theta+=np.outer(y-self.theta@phi,gain)
        self.P=(self.P-np.outer(gain,phi@self.P))/self.forgetting
        self.P=(self.P+self.P.T)/2
        tr=np.trace(self.P)
        if tr>self.max_trace: self.P*=self.max_trace/tr   # anti-windup under poor excitation
        if not (np.isfinite(self.theta).all() and np.isfinite(self.P).all()):
            raise FloatingPointError("RLS state became non-finite")


class OnlineCost:
    """Empirical completion-time/cost learning inside a precomputed winning set."""
    def __init__(self,alpha=.1):
        if not 0<alpha<=1: raise ValueError("Invalid learning rate")
        self.alpha=alpha; self.values={}; self.counts={}
    def update(self,name,cost):
        if not np.isfinite(cost) or cost<0: raise ValueError("Invalid cost")
        old=self.values.get(name,float(cost))
        self.values[name]=(1-self.alpha)*old+self.alpha*float(cost)
        self.counts[name]=self.counts.get(name,0)+1
