"""Numerical reference utilities. Velocity guards are not torque-level proofs."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def damped_ik(jacobian, twist, damping=.04):
    j=np.asarray(jacobian,float); v=np.asarray(twist,float)
    if (j.ndim!=2 or v.ndim!=1 or j.shape[0]!=len(v) or not np.isfinite(j).all() or
        not np.isfinite(v).all() or not np.isfinite(damping) or damping<=0):
        raise ValueError("Invalid IK inputs")
    return j.T@np.linalg.solve(j@j.T+damping*damping*np.eye(j.shape[0]),v)


def rotation_error(current_wxyz, desired_wxyz):
    def r(q):
        q=np.asarray(q,float)
        if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8: raise ValueError("Invalid quaternion")
        return Rotation.from_quat(q[[1,2,3,0]]/np.linalg.norm(q))
    return (r(desired_wxyz)*r(current_wxyz).inv()).as_rotvec()


def guarded_velocity(nominal,q,previous,lower,upper,velocity_limit,acceleration_limit,dt,
                     rows=None,rhs=None,margin=.03):
    """Intersection solve for a joint POSITION reference integrator.

    rows @ velocity >= rhs encodes supplied linearized guards. These are only
    as valid as the caller's reachability/tracking-error model. No safety slack.
    """
    nominal,q,previous=map(lambda x:np.asarray(x,float),(nominal,q,previous))
    if not np.isfinite([dt,margin]).all() or dt<=0 or margin<0: raise ValueError("Invalid timestep or margin")
    if q.ndim!=1 or nominal.shape!=q.shape or previous.shape!=q.shape:
        raise ValueError("Velocity input shape mismatch")
    if not all(np.isfinite(x).all() for x in [nominal,q,previous,lower,upper,velocity_limit,acceleration_limit]):
        raise ValueError("Nonfinite velocity inputs")
    if np.any(np.asarray(velocity_limit)<0) or np.any(np.asarray(acceleration_limit)<0):
        raise ValueError("Negative motion limit")
    a=np.broadcast_to(np.asarray(acceleration_limit,float),q.shape)
    room_up=np.maximum(0.0,np.asarray(upper)-margin-q)
    room_dn=np.maximum(0.0,q-np.asarray(lower)-margin)
    # Discrete-time braking (viability) bounds: v dt + v^2/(2a) <= distance to the
    # margin, so the reference can always stop before the limit (recursive feasibility).
    with np.errstate(divide='ignore',invalid='ignore'):
        brake_up=np.where(a>0,-a*dt+np.sqrt((a*dt)**2+2*a*room_up),room_up/dt)
        brake_dn=np.where(a>0,-a*dt+np.sqrt((a*dt)**2+2*a*room_dn),room_dn/dt)
    lo=np.maximum.reduce([-np.broadcast_to(velocity_limit,q.shape),
                          previous-a*dt,
                          (np.asarray(lower)+margin-q)/dt, -brake_dn])
    hi=np.minimum.reduce([np.broadcast_to(velocity_limit,q.shape),
                          previous+a*dt,
                          (np.asarray(upper)-margin-q)/dt, brake_up])
    # A reference that is already moving toward a limit faster than it can brake
    # (e.g. after an external reset) gets the maximal deceleration on THAT joint
    # instead of an empty interval; viable joints keep the exact intersection.
    stuck=lo>hi
    if np.any(stuck):
        decel=previous-np.sign(previous)*np.minimum(np.abs(previous),a*dt)
        lo=np.where(stuck,decel,lo); hi=np.where(stuck,decel,hi)
    if np.any(lo>hi): raise RuntimeError("Infeasible velocity/acceleration/position intersection")
    if rows is None or len(rows)==0:
        return np.clip(nominal,lo,hi)
    a,b=np.asarray(rows,float),np.asarray(rhs,float)
    if a.ndim!=2 or a.shape[1]!=len(q) or b.shape!=(a.shape[0],) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid linear guards')
    r=minimize(lambda x:.5*np.sum((x-nominal)**2),np.clip(nominal,lo,hi),
               jac=lambda x:x-nominal,bounds=list(zip(lo,hi)),method='SLSQP',
               constraints=[{'type':'ineq','fun':lambda x:a@x-b,'jac':lambda x:a}],
               options={'ftol':1e-12,'maxiter':100})
    if not r.success or np.min(a@r.x-b)<-1e-8:
        raise RuntimeError("Guard infeasible; do not send an unchecked nominal command")
    return r.x


def stopping_distance(speed,deceleration,latency=0.,margin=0.):
    if (not np.isfinite([speed,deceleration,latency,margin]).all() or
            min(speed,latency,margin)<0 or deceleration<=0): raise ValueError("Invalid braking data")
    return speed*latency+speed*speed/(2*deceleration)+margin
