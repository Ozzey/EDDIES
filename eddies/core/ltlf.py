"""Finite-trace LTL with exact finite-word derivatives and independent evaluator.

Syntax: !, &, |, ->, X, F, G, U, parentheses; identifiers are propositions.
Reserved words: X F G U true false nonempty (an identifier is read greedily, so
'GFa' is ONE proposition — write 'G F a').  Precedence (tightest first): unary
! X F G, then U, &, |, ->; U and -> are right-associative.  A letter is a SET of
proposition names (a bare string is rejected, never split into characters).
Pass ``propositions=`` to parse/Monitor to reject unknown/mistyped atoms.
X is STRONG next: X true is false on a one-letter trace. Nonempty-suffix
sentinel NE is essential; dropping it is a common finite-trace bug.
This is not an infinite-word Buchi synthesis implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import re
from itertools import product


@dataclass(frozen=True)
class Formula:
    op: str
    args: tuple = ()

    def __post_init__(self):
        # cached hash: residuals are hashed constantly (monitor, game states)
        object.__setattr__(self, "_hash", hash((self.op, self.args)))

    def __hash__(self):
        return self._hash

    def __str__(self):
        if self.op == 'atom': return self.args[0]
        if not self.args: return self.op
        return self.op+'('+','.join(map(str,self.args))+')'


TRUE, FALSE, NE = Formula('true'), Formula('false'), Formula('nonempty')


@lru_cache(maxsize=200000)
def _key(f: Formula) -> str:
    return str(f)


def _negate(f: Formula) -> Formula:
    """Negation pushed through & and | (De Morgan); literals keep a ! wrapper."""
    if f == TRUE: return FALSE
    if f == FALSE: return TRUE
    if f.op == '!': return f.args[0]
    if f.op == '&': return make('|', *(_negate(a) for a in f.args))
    if f.op == '|': return make('&', *(_negate(a) for a in f.args))
    return Formula('!', (f,))


def _clauses(f: Formula) -> frozenset:
    """DNF of a canonical formula: frozenset of frozensets of literals."""
    if f == TRUE: return frozenset([frozenset()])
    if f == FALSE: return frozenset()
    if f.op == '|':
        out = set()
        for a in f.args: out |= _clauses(a)
        return frozenset(out)
    if f.op == '&':
        out = {frozenset()}
        for a in f.args:
            ca = _clauses(a)
            out = {x | y for x in out for y in ca}
        return frozenset(out)
    return frozenset([frozenset([f])])


def _from_clauses(clauses) -> Formula:
    """Canonical DNF: drop contradictory clauses, absorb supersets, sort."""
    clauses = [c for c in set(clauses) if not any(_negate(l) in c for l in c)]
    minimal = [c for c in clauses if not any(o < c for o in clauses)]
    if not minimal: return FALSE
    if any(len(c) == 0 for c in minimal): return TRUE
    terms = []
    for c in minimal:
        lits = sorted(c, key=_key)
        terms.append(lits[0] if len(lits) == 1 else Formula('&', tuple(lits)))
    if len(terms) == 1: return terms[0]
    return Formula('|', tuple(sorted(terms, key=_key)))


def make(op, *args):
    """Smart constructor keeping every Boolean combination in canonical DNF.

    Boolean structure is normalized to a disjunction of conjunctions of literals
    (atoms, the NE sentinel, temporal subformulas and their negations) with
    contradiction removal and absorption (x | (x & y) = x).  Literals of the
    residuals of a formula are drawn from its finite closure, so the set of
    residuals reachable by progression is finite and progression of an X-free
    formula is idempotent under stuttering, so nested U derivatives cannot grow
    without bound.
    """
    if op == '!':
        f, = args
        return _negate(f)
    if op in ('&','|'):
        if op == '|':
            out = set()
            for a in args: out |= _clauses(a)
            return _from_clauses(out)
        out = {frozenset()}
        for a in args:
            ca = _clauses(a)
            out = {x | y for x in out for y in ca}
            if not out: return FALSE
        return _from_clauses(out)
    if op == 'F':
        if args[0] == FALSE: return FALSE
        if args[0].op == 'F': return args[0]
    if op == 'G':
        if args[0] == TRUE: return TRUE
        if args[0].op == 'G': return args[0]
    if op == 'U' and args[1] == FALSE: return FALSE
    return Formula(op,tuple(args))


RESERVED = {'X', 'F', 'G', 'U', 'true', 'false', 'nonempty'}


def _raw(op, *args):
    """Unsimplified constructor (used by the independent evaluator)."""
    if op == '!' and args[0] in (TRUE, FALSE): return FALSE if args[0] == TRUE else TRUE
    return Formula(op, tuple(args))


def parse(text: str, propositions=None, simplify: bool = True) -> Formula:
    """Parse an LTLf formula.  ``simplify=False`` returns the raw syntax tree
    (no make() rewriting) for an evaluator independent of the simplifier."""
    try:
        return _parse(text, propositions, make if simplify else _raw)
    except RecursionError:
        raise ValueError("LTLf formula nesting too deep") from None


def _parse(text, propositions, mk) -> Formula:
    pattern = r'\s*(->|[()!&|]|[A-Za-z_][A-Za-z_0-9]*)'
    tokens=[]; pos=0
    while pos<len(text):
        if not text[pos:].strip(): break
        match=re.match(pattern,text[pos:])
        if not match: raise ValueError(f"Invalid LTLf syntax at {text[pos:]!r}")
        tokens.append(match.group(1)); pos+=match.end()
    i=0
    def expr(min_prec=0):
        nonlocal i
        if i>=len(tokens): raise ValueError("Missing operand")
        t=tokens[i]; i+=1
        if t in ('!','X','F','G'):
            lhs=mk(t,expr(5))
        elif t=='(':
            lhs=expr()
            if i>=len(tokens) or tokens[i]!=')': raise ValueError("Unclosed parenthesis")
            i+=1
        elif t in ('true','false'):
            lhs=TRUE if t=='true' else FALSE
        elif re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',t) and t not in RESERVED:
            if propositions is not None and t not in propositions:
                raise ValueError(f"Unknown proposition {t!r}")
            lhs=Formula('atom',(t,))
        else: raise ValueError(f"Unexpected token {t!r}")
        precedences={'->':1,'|':2,'&':3,'U':4}
        while i<len(tokens) and tokens[i] in precedences and precedences[tokens[i]]>=min_prec:
            op=tokens[i]; i+=1; prec=precedences[op]
            rhs=expr(prec if op in ('->','U') else prec+1)
            lhs=mk('|',mk('!',lhs),rhs) if op=='->' else mk(op,lhs,rhs)
        return lhs
    result=expr()
    if i!=len(tokens): raise ValueError("Trailing LTLf tokens")
    return result


@lru_cache(maxsize=100000)
def nullable(f: Formula) -> bool:
    """Acceptance of the empty residual, NOT permission for an empty episode."""
    op,a=f.op,f.args
    if op in ('false','atom','nonempty','X','F','U'): return False
    if op in ('true','G'): return True
    if op=='!': return not nullable(a[0])
    if op=='&': return all(map(nullable,a))
    if op=='|': return any(map(nullable,a))
    raise ValueError(op)


@lru_cache(maxsize=100000)
def progress(f: Formula, letter: frozenset[str]) -> Formula:
    op,a=f.op,f.args
    if op in ('true','false'): return f
    if op=='nonempty': return TRUE
    if op=='atom': return TRUE if a[0] in letter else FALSE
    if op in ('!','&','|'): return make(op,*(progress(g,letter) for g in a))
    if op=='X': return make('&',NE,a[0])
    if op=='F': return make('|',progress(a[0],letter),f)
    if op=='G': return make('&',progress(a[0],letter),f)
    if op=='U': return make('|',progress(a[1],letter),make('&',progress(a[0],letter),f))
    raise ValueError(op)


def _letter(x) -> frozenset:
    if isinstance(x, str):
        raise TypeError("a letter is a set of proposition names, not a string")
    return frozenset(x)


def consume(f, trace):
    for letter in trace: f=progress(f,_letter(letter))
    return f


def atoms(f):
    if f.op=='atom': return {f.args[0]}
    return set().union(*(atoms(a) for a in f.args)) if f.args else set()


@lru_cache(maxsize=20000)
def dead(f: Formula, cap: int = 4096) -> bool:
    """True iff NO continuation (not even the empty one) can satisfy residual f.

    Exact BFS over the finite residual closure for the letters over f's atoms
    (2^|atoms| letters); returns False ('not yet refuted') if the cap is hit.
    """
    if f == FALSE: return True
    names = sorted(atoms(f))
    if len(names) > 10: return False
    letters = [frozenset(n for n, b in zip(names, bits) if b)
               for bits in product((False, True), repeat=len(names))]
    seen, frontier = {f}, [f]
    while frontier:
        g = frontier.pop()
        if nullable(g): return False
        for letter in letters:
            h = progress(g, letter)
            if h != FALSE and h not in seen:
                seen.add(h)
                if len(seen) > cap: return False
                frontier.append(h)
    return True


class Monitor:
    def __init__(self, formula: str | Formula, propositions=None):
        self.initial=parse(formula, propositions) if isinstance(formula,str) else formula
        self.state=self.initial; self.steps=0
    def step(self, true_propositions):
        self.state=progress(self.state,_letter(true_propositions)); self.steps+=1
        return self.status
    @property
    def status(self):
        """'violated' once no finite continuation can satisfy the formula
        (syntactic FALSE or a semantically dead residual); otherwise
        'accepting_if_terminated' or 'pending'."""
        if self.state==FALSE or dead(self.state): return 'violated'
        if self.steps>0 and nullable(self.state): return 'accepting_if_terminated'
        return 'pending'
    def accepts(self): return self.steps>0 and nullable(self.state)


def evaluate(formula, trace):
    """Direct memoized finite-trace semantics (quadratic worst case in length).

    Given a STRING it parses with simplify=False, so the result is independent of
    both progression and the make() simplifier; given a Formula it evaluates it as is.
    """
    f=parse(formula, simplify=False) if isinstance(formula,str) else formula
    word=tuple(_letter(s) for s in trace)
    if not word: return False
    @lru_cache(None)
    def at(g,i):
        op,a=g.op,g.args
        if op=='true': return True
        if op=='false': return False
        if op=='nonempty': return i<len(word)
        if op=='atom': return a[0] in word[i]
        if op=='!': return not at(a[0],i)
        if op=='&': return all(at(x,i) for x in a)
        if op=='|': return any(at(x,i) for x in a)
        if op=='X': return i+1<len(word) and at(a[0],i+1)
        if op=='F': return any(at(a[0],j) for j in range(i,len(word)))
        if op=='G': return all(at(a[0],j) for j in range(i,len(word)))
        if op=='U':
            return any(at(a[1],j) and all(at(a[0],k) for k in range(i,j)) for j in range(i,len(word)))
        raise ValueError(op)
    return at(f,0)


def alphabet(propositions):
    names=sorted(propositions)
    return [frozenset(n for n,b in zip(names,bits) if b)
            for bits in product((False,True),repeat=len(names))]
