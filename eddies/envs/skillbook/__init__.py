"""The skillbook: ONE vocabulary of action tokens -- Eddies -- shared by every task.

An Eddie (EDDIE: Embedding-space Discrete Dynamical Instruction with Envelope) is a primitive
with arguments (``reach:peg_above`` = the ``reach`` primitive bound to the frame "10 cm above
the peg").  The primitive's behaviour (target generation, outcome
detection, timeouts) is implemented once in :mod:`primitives`; a task only supplies
perception frames and object views and composes Eddies in its skill graph.  Certificates
are learned per primitive (pooled over every task and binding that uses it), so the
vocabulary can grow: adding a primitive requires certifying that primitive and its new
hand-overs only.

Version 1 vocabulary (11 primitives, following the standard manipulation skill taxonomies:
Bogh et al. ISR 2012; MAPLE, Nasiriany et al. 2021; RAPS, Dalal et al. NeurIPS 2021; FMB):
    reach, grasp, lift, transport, place, release, insert, push, pull, turn, retreat
"""
from eddies.envs.skillbook.book import BOOK, SkillBook, primitive_of  # noqa: F401
