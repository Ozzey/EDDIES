"""Composition of Eddies from LTLf specifications (eddies.core.compose + skillbook contracts)."""
import pytest

from eddies.common.paths import MENAGERIE
from eddies.core.compose import Effect, Schema, cell_name, compose, lit, outcome_word, parse_cell

# the task-level tests build MuJoCo scenes from the menagerie checkout (gitignored; scripts/download_menagerie.sh)
needs_menagerie = pytest.mark.skipif(not MENAGERIE.exists(), reason="external/mujoco_menagerie missing (scripts/download_menagerie.sh)")


def test_effect_and_cells():
    e = Effect(add=frozenset({"b"}), delete=frozenset({"a"}), delete_prefix=("x_",), cond_add=(("c", "d"),))
    assert e.apply(frozenset({"a", "c", "x_1"})) == frozenset({"b", "c", "d"})
    assert parse_cell(cell_name({"q", "p"})) == frozenset({"p", "q"}) and cell_name(()) == "none"
    assert outcome_word(frozenset({"a", "b"}), frozenset({"b", "c"})) == (frozenset({"b"}), frozenset({"b", "c"}))


def test_closure_and_failures():
    pick = Schema("pick", ((lit("!held"),),), Effect(add=frozenset({"held"})),
                  failures=(("missed", Effect()),))
    drop = Schema("drop", ((lit("held"),),), Effect(add=frozenset({"placed"}), delete=frozenset({"held"})))
    trs, states = compose([pick, drop], [frozenset()])
    assert {t.eddie for t in trs} == {"pick", "drop"}
    assert frozenset({"placed"}) in states
    p0 = next(t for t in trs if t.eddie == "pick" and t.source == frozenset())
    assert [o[0] for o in p0.outcomes] == ["ok", "missed"] and p0.outcomes[1][3]


HAND = {"franka_cube_stack": "franka_cube_stack_ltl", "franka_peg_insertion": "franka_peg_insertion_ltl",
        "franka_drawer": "franka_drawer_ltl", "franka_fmb": "franka_fmb_ltl"}


@needs_menagerie
@pytest.mark.parametrize("hand", list(HAND))
def test_composed_plan_equals_hand_written(hand):
    from eddies.envs.tasks import make_task
    a, b = make_task(hand, 2000), make_task(HAND[hand], 2000)
    assert b.nominal_plan() == a.nominal_plan()
    sa, sb = a.skills(), b.skills()
    assert all(abs(sa[n].delta - sb[n].delta) < 1e-12 for n in sa)       # same certified Eddies


@pytest.mark.parametrize("task,spec", [
    ("franka_drawer_ltl", "G !bad & G !open_drawer & F(on_box_goal)"),
    ("franka_cube_stack_ltl", "G !bad & G !holding_red & F(on_red_blue)"),
])
@needs_menagerie
def test_unrealisable_spec_refused(task, spec):
    from eddies.envs.tasks import make_task
    with pytest.raises(ValueError, match="not realisable"):
        make_task(task, 2000, spec=spec)


@needs_menagerie
def test_unknown_atom_rejected():
    from eddies.envs.tasks import make_task
    with pytest.raises(ValueError, match="not fluents"):
        make_task("franka_cube_stack_ltl", 2000, spec="F(on_blue_red)")
