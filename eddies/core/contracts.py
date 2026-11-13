"""Finite AND-OR reachability game over (cell, previous token, LTLf residual).

A Contract is trusted input: the code cannot prove a robot tube from a name.
Each outcome contains a COMPLETE primitive-step label word, not only an endpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
from .ltlf import Formula, parse, nullable, consume, FALSE, dead


@dataclass(frozen=True)
class Outcome:
    target: str
    labels: tuple[frozenset[str], ...]
    def __post_init__(self):
        if not self.labels: raise ValueError("A token must have at least one primitive step")


@dataclass(frozen=True)
class Contract:
    name: str
    source: str
    embedding: tuple[float,...]
    outcomes: tuple[Outcome,...]
    cost: float=1.0
    certified: bool=False
    certificate_id: str=''
    def __post_init__(self):
        if (self.name=='@initial' or not self.outcomes or np.asarray(self.embedding).ndim!=1 or
            not len(self.embedding) or not np.isfinite(self.embedding).all() or
            not np.isfinite(self.cost) or self.cost<0):
            raise ValueError("Nonempty outcomes and finite embedding required")
        if self.certified and not self.certificate_id:
            raise ValueError("A certificate identifier is required")


@dataclass(frozen=True)
class State:
    cell: str
    previous: str
    residual: Formula


class ContractGame:
    def __init__(self, contracts, formula, initial_cell, initial_embedding, rho,
                 terminal_cells, metric=None, require_certificates=True, max_states=20000,
                 initial_labels=None, propositions=None):
        # sigma_0 must be OBSERVED and passed explicitly, never fabricated
        if initial_labels is None:
            raise ValueError("initial_labels (the observed first letter) is required")
        if isinstance(initial_labels, str) or isinstance(terminal_cells, str):
            raise TypeError("initial_labels / terminal_cells must be collections, not strings")
        self.contracts={c.name:c for c in contracts}
        if len(self.contracts)!=len(contracts): raise ValueError("Duplicate token names")
        if require_certificates and not all(c.certified for c in contracts):
            raise ValueError("Uncertified contract: physical guarantees are unavailable")
        self.initial_embedding=np.asarray(initial_embedding,float)
        self.metric=np.eye(len(initial_embedding)) if metric is None else np.asarray(metric,float)
        if not np.isfinite(rho) or rho < 0: raise ValueError("Invalid slew radius")
        if self.initial_embedding.ndim!=1 or not np.isfinite(self.initial_embedding).all():
            raise ValueError("Invalid initial embedding")
        n=len(initial_embedding)
        if self.metric.shape!=(n,n) or not np.isfinite(self.metric).all() or not np.allclose(self.metric,self.metric.T):
            raise ValueError("Invalid metric")
        np.linalg.cholesky(self.metric)
        if any(len(c.embedding)!=len(initial_embedding) for c in contracts):
            raise ValueError("Embedding dimension mismatch")
        self.rho=rho; self.terminals=set(terminal_cells)
        self.initial=State(initial_cell,'@initial',consume(parse(formula, propositions), [initial_labels]))
        self.edges={}; self.rank={}; self.max_states=max_states
        self._expand(); self._solve()

    def _expand(self):
        queue=deque([self.initial]); seen={self.initial}
        while queue:
            s=queue.popleft(); edges=[]
            prev=(self.initial_embedding if s.previous=='@initial'
                  else np.asarray(self.contracts[s.previous].embedding))
            accepting = s.cell in self.terminals and nullable(s.residual)
            if s.residual!=FALSE and not dead(s.residual) and not accepting:   # rank 0 needs no expansion
                for c in self.contracts.values():
                    d=np.asarray(c.embedding)-prev
                    if c.source!=s.cell or not (d@self.metric@d<=self.rho**2*(1+1e-12)): continue
                    children=tuple(State(o.target,c.name,consume(s.residual,o.labels)) for o in c.outcomes)
                    edges.append((c.name,children))
                    for child in children:
                        if child not in seen:
                            seen.add(child); queue.append(child)
                            if len(seen)>self.max_states: raise RuntimeError("Product state cap exceeded")
            self.edges[s]=edges

    def _solve(self):
        self.rank={s:0 for s in self.edges if s.cell in self.terminals and nullable(s.residual)}
        k=0
        while True:
            added={s:k+1 for s,edges in self.edges.items() if s not in self.rank
                   and any(all(t in self.rank for t in children) for _,children in edges)}
            if not added: break
            self.rank.update(added); k+=1

    def allowed(self, state):
        if state not in self.rank or self.rank[state]==0: return []
        return [name for name,children in self.edges[state]
                if all(t in self.rank and self.rank[t]<self.rank[state] for t in children)]

    def choose(self,state,costs=None):
        choices=self.allowed(state)
        if not choices: raise RuntimeError("No progress-preserving action (or terminate)")
        costs={} if costs is None else costs
        return min(choices,key=lambda name:(costs.get(name,self.contracts[name].cost),name))

    def successor(self,state,name,outcome_index):
        if name not in self.allowed(state): raise ValueError("Action not permitted by winning policy")
        if type(outcome_index) is not int or outcome_index<0: raise IndexError("Invalid outcome index")
        for n,children in self.edges[state]:
            if n==name: return children[outcome_index]
        raise KeyError(name)
