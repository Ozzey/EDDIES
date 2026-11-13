import itertools
import numpy as np
import pytest
from eddies.core.certificates import Ellipsoid, AffinePhase, compose, composed_error, decoder_slew_bound
from eddies.core.embedding import EmbeddingProjector, InfeasibleEmbedding, discrete_neighbors
from eddies.core.ltlf import parse, consume, nullable, evaluate, alphabet, Monitor, progress
from eddies.core.contracts import Outcome, Contract, ContractGame
from eddies.core.online import SetMembershipModel, ModelInconsistency, RecursiveLeastSquares
from eddies.core.control import rotation_error, damped_ik, guarded_velocity, stopping_distance
from eddies.core.metrics import wilson, zero_failure_upper

@pytest.mark.parametrize('seed',range(5))
def test_ellipsoid_perturbation(seed):
    rng=np.random.default_rng(seed)
    B=rng.normal(size=(3,3)); P=B.T@B+np.eye(3)
    C=Ellipsoid(np.ones(3)*.3,P,1.)
    phase=AffinePhase(np.eye(3)*.65,np.zeros(3),.05,.1,.1)
    delta=phase.admissible_radius(C)*.95
    assert phase.margin(C,delta)>=0
    for _ in range(100):
        x=rng.normal(size=3); x=x/C.norm(x)*rng.random()
        A=rng.normal(size=(3,3)); A=A/C.operator_norm(A)*.1*delta
        b=rng.normal(size=3); b=b/C.norm(b)*.1*delta
        w=rng.normal(size=3); w=w/C.norm(w)*.05
        nxt=C.center+(phase.A+A)@x+b+w
        assert C.contains(nxt,1e-10)

def test_composition():
    a=(np.array([[.3]]),np.array([.2])); b=(np.array([[.5]]),np.array([.1]))
    A,B=compose(a,b)
    assert np.allclose(A,[[.15]]) and np.allclose(B,[.2])
    assert composed_error([.5,.4],[.1,.2])==pytest.approx(.24)

def test_bad_certificate():
    with pytest.raises(ValueError): Ellipsoid(np.zeros(2),np.diag([1,-1]),1)
    C=Ellipsoid(np.zeros(1),np.eye(1),1)
    with pytest.raises(ValueError): AffinePhase(np.eye(1)*2,np.zeros(1),0).admissible_radius(C)

def test_project_intersection():
    p=EmbeddingProjector(np.array([-1.,-1]),np.array([1.,1]),np.diag([1.,2.]))
    e=p.project([2.,2.],[0.,0.],.3,anchor=[.1,.1],delta=.25)
    assert p.norm(e)<=.3+1e-8 and p.norm(e-[.1,.1])<=.25+1e-8
    with pytest.raises(InfeasibleEmbedding): p.project([1,1],[-1,-1],.1,[1,1],.1)
    with pytest.raises(ValueError): p.project([np.nan,0],[0,0],.1)
    assert np.allclose(p.project([1,1],[0,0],0),[0,0])
    assert len(discrete_neighbors(np.array([[-1],[0],[1]]),[0],.1))==1

FORMULAS=['a','!a','X true','X a','!X a','G a','F a','a U b',
          'G(a -> F b)','F(a & F b)','G !b & F a', 'X G a',
          '!(a U b)','G F a','F G b','a U X b','G(X a | b)',
          'X X true','F X a','G(a | X b)','false','true','X !a',
          '!G X(a & b)','(a U b) U a','a U (b U a)']
@pytest.mark.parametrize('formula',FORMULAS)
def test_finite_logic_exhaustive(formula):
    # evaluate(str) parses WITHOUT the make() simplifier: independent of both
    # progression and simplification.
    f=parse(formula); letters=alphabet(['a','b'])
    for length in range(1,5):
        for word in itertools.product(letters,repeat=length):
            assert nullable(consume(f,word))==evaluate(formula,word),(formula,word)

def test_boundary_logic():
    assert not evaluate('X true',[set()])
    m=Monitor('G !bad & F done')
    assert m.step({'done'})=='accepting_if_terminated'
    assert m.step({'bad'})=='violated'
    assert not m.accepts()
    for invalid in ['', '(a', 'a b', 'a + b']:
        with pytest.raises(ValueError): parse(invalid)

def contract(name,source,e,outcomes):
    return Contract(name,source,(e,),tuple(Outcome(t,tuple(frozenset(x) for x in w)) for t,w in outcomes),
                    certified=True,certificate_id='abstract-exact-fixture-v1')

def game(rho=1):
    cs=[contract('go_a','start',.3,[('middle',[{'a'}])]),
        contract('go_b','middle',.6,[('end',[{'b'}])]),
        contract('risky','start',.3,[('end',[{'a'},{'b'}]),('end',[{'bad'},{'a'},{'b'}])]),
        contract('stall','start',0,[('start',[set()])])]
    return ContractGame(cs,'G !bad & F(a & F b)','start',[0],rho,{'end'},initial_labels=frozenset())

def test_robust_game_rank_and_all_outcomes():
    g=game(); s=g.initial
    assert g.rank[s]==2 and g.allowed(s)==['go_a']
    s=g.successor(s,'go_a',0); assert g.choose(s)=='go_b'
    s=g.successor(s,'go_b',0); assert g.rank[s]==0
    assert g.allowed(s)==[]
    with pytest.raises(ValueError): g.successor(g.initial,'risky',0)

def test_slew_disconnect():
    g=game(.1); assert g.initial not in g.rank

def test_initial_label_is_observed():
    g=ContractGame([], 'G !bad','end',[0],1,{'end'},initial_labels={'bad'})
    assert g.initial not in g.rank
    h=ContractGame([], 'F a','end',[0],1,{'end'},initial_labels={'a'})
    assert h.rank[h.initial]==0

def test_uncertified_contract_rejected():
    c=Contract('unsafe','x',(0,), (Outcome('x',(frozenset(),)),))
    with pytest.raises(ValueError): ContractGame([c],'F a','x',[0],1,{'x'},initial_labels=frozenset())

def test_online_nested_and_true_containment():
    m=SetMembershipModel([[-1,-1]],[[1,1]],.02)
    rng=np.random.default_rng(7); theta=np.array([.7,.2]); query=np.array([.4,.9])
    prevlo,prevhi=m.predict_interval(query)
    for _ in range(25):
        phi=rng.uniform(-1,1,2); y=np.array([phi@theta+rng.uniform(-.02,.02)])
        m.update(phi,y); lo,hi=m.predict_interval(query)
        assert lo<=theta@query<=hi and lo>=prevlo-1e-8 and hi<=prevhi+1e-8
        prevlo,prevhi=lo,hi
    version=m.version
    with pytest.raises(ModelInconsistency): m.update([0,0],[100])
    assert m.version==version

def test_rls():
    r=RecursiveLeastSquares(2,1,ridge=.0001)
    for _ in range(10):
        r.update([1,0],[.7]); r.update([0,1],[.2])
    assert np.allclose(r.theta,[[.7,.2]],atol=1e-4)

def test_rotation_and_ik():
    q=np.array([0.,1,0,0]); target=-q
    assert np.linalg.norm(rotation_error(target,q))<1e-10
    assert np.allclose(damped_ik(np.eye(6),np.ones(6),.1),np.ones(6)/1.01)

def test_guarded_velocity():
    import inspect
    # Match public signature explicitly; no untested robot dynamics premise here.
    v=guarded_velocity(np.array([10.]),np.array([0.]),np.array([0.]),
                       np.array([-1.]),np.array([1.]),np.array([1.]),np.array([2.]),.1)
    assert abs(v[0])<=.2+1e-8
    assert stopping_distance(1.,2.,.1,.01)==pytest.approx(.36)

def test_stats():
    low,high=wilson(5,10)
    assert low==pytest.approx(.2366,abs=1e-4) and high==pytest.approx(.7634,abs=1e-4)
    assert zero_failure_upper(100)==pytest.approx(.029513,abs=1e-6)

@pytest.mark.parametrize('k',[500,1000,2500,4000])
def test_local_contact_lyapunov(k):
    from eddies.core.contact import contact_certificate
    result=contact_certificate(k,alpha=1e-4,beta=.2)
    assert result['spectral_radius']<1 and result['contraction']<1
    assert np.allclose(result['A'].T@result['P']@result['A']-result['P'],-np.eye(2))

def test_contact_unstable_gain_rejected():
    from eddies.core.contact import contact_certificate
    with pytest.raises(ValueError): contact_certificate(2500,alpha=.001,beta=.2)

def test_uncertain_labels_branch_conservatively():
    from eddies.core.labels import possible_letters
    letters=possible_letters({'safe':(.1,.2),'goal':(-.1,.2),'bad':(-2,-1)})
    assert set(letters)=={frozenset({'safe'}),frozenset({'safe','goal'})}
    with pytest.raises(ValueError): possible_letters({'x':(2,1)})

def test_rotation_error_has_correct_feedback_sign():
    from scipy.spatial.transform import Rotation
    identity=np.array([1.,0,0,0])
    desired=Rotation.from_euler('z',.3).as_quat()[[3,0,1,2]]
    assert np.allclose(rotation_error(identity,desired),[0,0,.3])
    assert np.allclose(rotation_error(desired,identity),[0,0,-.3])

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.])
def test_nonfinite_certificate_and_game_inputs_rejected(bad):
    with pytest.raises(ValueError): Ellipsoid(np.zeros(1),np.eye(1),bad)
    with pytest.raises(ValueError): AffinePhase(np.eye(1),np.zeros(1),bad)
    with pytest.raises(ValueError): ContractGame([], 'F a','end',[0],bad,{'end'},initial_labels=frozenset())

# ---------------------------------------------------------------------------
# Regression tests: projector and monitor edge cases.
# ---------------------------------------------------------------------------
def test_projector_counterexamples_and_gauge():
    P=EmbeddingProjector([-1.],[1.],[[1.]])
    assert P.project([-1.1714685612925249],[-0.32],0.08)==pytest.approx([-0.40])
    P8=EmbeddingProjector(np.r_[-np.ones(6),0,0],np.ones(8),np.eye(8))
    prev=np.array([0.1334,-0.4202,-0.93,0,0,0,0.29499999955229783,0]); prop=prev.copy(); prop[6]=0
    assert P8.project(prop,prev,0.015)[6]==pytest.approx(0.28,abs=1e-8)
    ref=None
    for s in [1,1e2,1e4,1e6,1e8]:
        x=EmbeddingProjector(-np.ones(3),np.ones(3),s*np.eye(3)).project([0.9,0.8,-0.7],[0.2,-0.1,0.3],0.25*np.sqrt(s))
        ref=x if ref is None else ref
        assert np.allclose(x,ref,atol=1e-9)
    assert np.allclose(ref,[0.315392,0.048361,0.135155],atol=1e-6)

@pytest.mark.parametrize('seed',range(4))
def test_projector_exact_against_dense_sampling(seed):
    rng=np.random.default_rng(seed)
    for _ in range(25):
        lo,hi=-np.ones(2),np.ones(2); M=np.diag(rng.uniform(.2,3,2))
        P=EmbeddingProjector(lo,hi,M)
        prev=rng.uniform(-.8,.8,2); rho=rng.uniform(.05,.6); p=prev+rng.normal(size=2)
        anchor=rng.uniform(-.8,.8,2); delta=max(.05,P.norm(anchor-prev)-rho+.5*rho)
        x=P.project(p,prev,rho,anchor,delta)             # never raises: prev-anchor segment is feasible
        assert P.norm(x-prev)<=rho+1e-8 and P.norm(x-anchor)<=delta+1e-8
        g=np.stack(np.meshgrid(np.linspace(-1,1,401),np.linspace(-1,1,401)),-1).reshape(-1,2)
        d1=np.einsum('bi,ij,bj->b',g-prev,M,g-prev); d2=np.einsum('bi,ij,bj->b',g-anchor,M,g-anchor)
        feas=g[(d1<=rho**2)&(d2<=delta**2)]
        if len(feas):
            best=np.min(np.einsum('bi,ij,bj->b',feas-p,M,feas-p))
            assert (x-p)@M@(x-p)<=best+1e-9

def test_ltlf_residuals_finite_idempotent_and_anticipatory():
    f=parse('(F a) U (F b)')
    seen={f}; frontier=[f]
    while frontier:
        g=frontier.pop()
        for letter in alphabet(['a','b']):
            h=progress(g,letter)
            if h not in seen: seen.add(h); frontier.append(h)
        assert len(seen)<50
    m=Monitor('(G safe) U (F done)')
    for _ in range(10000): m.step({'safe'})
    assert m.status=='pending'
    for text in ['G !bad & F(a & F b)','(G b) U (G a)','(F(F c)) U G a','(true U a) U F b']:
        f=parse(text)
        for letter in alphabet(['a','b','c','bad']):
            g=progress(f,letter)
            assert progress(g,letter)==g
    m=Monitor('G X true'); assert m.step(set())=='violated'
    m=Monitor('G a & F !a'); m.step({'a'}); assert m.status=='violated'

def test_ltlf_input_hardening():
    with pytest.raises(TypeError): Monitor('F done').step('done')
    with pytest.raises(ValueError): parse('F grasp & G !bad',propositions={'grasped','bad'})
    with pytest.raises(ValueError): parse('X'*5000+' a')
    assert evaluate('G(a | !a)',[set()]) and not evaluate('X true',[set()])

def test_guarded_velocity_is_recursively_feasible():
    q=np.array([2.]); v=np.zeros(1)
    for _ in range(3000):
        v=guarded_velocity(np.array([10.]),q,v,np.array([-2.8973]),np.array([2.8973]),
                           np.array([.45]),np.array([1.]),4/240)
        q=q+v*4/240
        assert q[0]<=2.8973-.03+1e-9
    assert abs(q[0]-(2.8973-.03))<1e-6 and abs(v[0])<1e-9

def test_contact_certificate_near_marginal_is_sound():
    from eddies.core.contact import contact_certificate
    for gap in [1e-2,1e-4,1e-6,1e-8]:
        try: r=contact_certificate(2500,alpha=(1-gap)/2500,beta=.2)
        except RuntimeError: continue                     # refusing is sound
        assert r['contraction']<1
        A,P=r['A'],r['P']; assert np.allclose(P,P.T); assert np.linalg.eigvalsh(A.T@P@A-P).max()<0

def test_online_models_validate_and_do_not_wind_up():
    r=RecursiveLeastSquares(2,3)
    with pytest.raises(ValueError): r.update([1,0],[5.])
    w=RecursiveLeastSquares(2,1,ridge=1.,forgetting=.9)
    for _ in range(20000): w.update([1.,0.],[.3])
    assert np.isfinite(w.P).all() and np.isfinite(w.theta).all()
    with pytest.raises(ValueError): SetMembershipModel([[np.nan,0]],[[1,1]],.1)
    with pytest.raises(ValueError): RecursiveLeastSquares(2,1,ridge=np.nan)

def test_certificate_radius_rounding_and_validation():
    rng=np.random.default_rng(3)
    for _ in range(300):
        B=rng.normal(size=(3,3)); C=Ellipsoid(np.zeros(3),B@B.T+np.eye(3),rng.uniform(.5,2))
        ph=AffinePhase(np.eye(3)*rng.uniform(.1,.8),np.zeros(3),rng.uniform(0,.1),rng.uniform(.01,1),rng.uniform(.01,1))
        try: d=ph.admissible_radius(C)
        except ValueError: continue
        assert ph.margin(C,d)>=0
    C=Ellipsoid(np.zeros(1),np.eye(1),1.)
    flat=AffinePhase(np.eye(1)*.5,np.zeros(1),.1)
    assert flat.admissible_radius(C)==np.inf and flat.margin(C,np.inf)==pytest.approx(.4)
    with pytest.raises(ValueError): compose((np.eye(2),np.zeros((2,1))),(np.eye(2),np.zeros(2)))
    with pytest.raises(ValueError): Ellipsoid(np.zeros(2),np.diag([1.,1e-12]),1.)
    with pytest.raises(ValueError): Ellipsoid(np.zeros(2),np.array([[1.,.5],[0.,1.]]),1.)


def test_game_requires_observed_initial_letter_and_rejects_strings():
    with pytest.raises(ValueError): ContractGame([], 'F a','end',[0],1,{'end'})
    with pytest.raises(TypeError): ContractGame([], 'F a','end',[0],1,'end',initial_labels=frozenset())
    with pytest.raises(ValueError): ContractGame([], 'G !bda','end',[0],1,{'end'},initial_labels=frozenset(),propositions={'bad'})
    g=game(); s=g.initial
    with pytest.raises(IndexError): g.successor(s,'go_a',-1)
