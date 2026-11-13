"""eddies.evaluate regenerates the paper's tables from results/ when the result files are present.

Each step is skipped (not failed) when its input is missing, so a fresh clone without results/ passes;
with the released results/ every table must print and every figure must be written."""
import io
import contextlib

import pytest

from eddies.common.paths import FIGURES, ROOT

ev = pytest.importorskip("eddies.evaluate")

TABLES = ["table2", "table3", "tableD1", "tableD2", "tableD3", "tableD4", "tableD5"]


def _run(step, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ev.main([step, "--strict", *args])
    assert rc == 0
    return buf.getvalue()


@pytest.mark.parametrize("step", TABLES)
def test_table(step):
    if not ev.have(*ev.NEEDS[step]):
        pytest.skip(f"results/{ev.NEEDS[step][0]} not present")
    out = _run(step)
    assert ("|" in out or step == "table3") and "FAILED" not in out


def test_table1_without_bootstrap():
    if not ev.have(*ev.NEEDS["table1"]):
        pytest.skip("results/table1 not present")
    out = _run("table1", "--B", "0")
    assert "Eddies" in out and "FAILED" not in out


def test_tableD1_totals_match_paper():
    """Table D.1 'All' row: 5 violations, 890 finished (fixed), 1968 (one-step, from results/gate_margin/summary.json
    or the per-step logs), 1955 (self-calibrating), 1581 / 1627 caught."""
    if not ev.have("embodiments/embodiments.json", "adaptive/adaptive_aero_peg_tighten_sb_known_aero.json"):
        pytest.skip("results/embodiments or adaptive not present")
    out = _run("tableD1")
    last = [l for l in out.splitlines() if l.startswith("| All")][-1]
    cells = [c.strip() for c in last.strip("|").split("|")]
    assert cells[4] == "5" and cells[6] == "890" and cells[8] == "1955" and cells[9] == "1581 / 1627"
    have_gm = (ROOT / "gate_margin" / "summary.json").exists() or any((ROOT / "gate_margin").glob("*.npz"))
    assert cells[7] == ("1968" if have_gm else "--")


def test_pooled_gates_match_paper():
    """App. D pooled gates (from results/gate_margin/summary.json without the per-step logs): at 3 cm the
    self-calibrating gate stops 387 of the 395 runs that would leave mu and 1,328 finish."""
    if not ev.have(*ev.NEEDS["pooled"]):
        pytest.skip("results/gate_margin/summary.json not present")
    out = _run("pooled")
    i = out.index("=== mu = 3 cm")
    row = [l for l in out[i:].splitlines() if l.strip().startswith("a self-cal gate")][0].split()
    assert row[3:9] == ["2000", "395", "387", "8", "277", "1328"]


def test_table3_mujoco_row_matches_paper():
    """Table 3 MuJoCo row: fixed gate 89 / 11, caught 298 / 298."""
    if not ev.have("compare/stack_mujoco.json"):
        pytest.skip("results/compare not present")
    out = _run("table3")
    assert "89" in out and "298" in out
    assert "30 / 20" in out and "43 / 7" in out          # robot rows (real/table3_robot_rows.json)


@pytest.mark.parametrize("step", ["fig4", "fig5", "fig7"])
def test_figure(step):
    pytest.importorskip("matplotlib")
    if not ev.have(*ev.NEEDS[step]):
        pytest.skip(f"results/{ev.NEEDS[step][0]} not present")
    _run(step)
    name = {"fig4": "res_cert", "fig5": "res_gate", "fig7": "res_heldout"}[step]
    assert (FIGURES / f"{name}.pdf").exists(), f"{name}.pdf not written to {FIGURES}"


def test_results_root_is_announced():
    assert ROOT.name  # resolved from eddies/config.yaml paths.results
