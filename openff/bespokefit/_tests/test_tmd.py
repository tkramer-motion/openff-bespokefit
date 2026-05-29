"""Tests for the standalone OpenFF -> tmd (Timemachine) force field helpers.

The core merge/serialisation logic is exercised with lightweight stubs so it runs
without the OpenFF toolkit; a couple of toolkit-gated tests check the real unit
conversion against an installed OpenFF force field.
"""

import ast

import pytest

from openff.bespokefit._tmd import (
    collect_fitted_torsions,
    load_tmd_force_field,
    merge_torsions_into_tmd,
    proper_torsion_to_tmd_pattern,
    save_tmd_force_field,
    tmd_proper_torsion_smirks,
)


# --- stubs that duck-type the bits of OpenFF objects the helpers touch ---------------


class _Quantity:
    """A stand-in for an openff.units Quantity; ``m_as`` returns the raw magnitude."""

    def __init__(self, magnitude):
        self._magnitude = magnitude

    def m_as(self, _unit):
        return self._magnitude


class _Torsion:
    def __init__(self, smirks, terms, idivf=None):
        # terms: list of (k, phase, period)
        self.smirks = smirks
        for i, (k, phase, period) in enumerate(terms, start=1):
            setattr(self, f"k{i}", _Quantity(k))
            setattr(self, f"phase{i}", _Quantity(phase))
            setattr(self, f"periodicity{i}", period)
            if idivf is not None:
                setattr(self, f"idivf{i}", idivf)


class _Handler:
    def __init__(self, parameters):
        self.parameters = parameters


class _ForceField:
    def __init__(self, parameters):
        self._handler = _Handler(parameters)

    def get_parameter_handler(self, _name):
        return self._handler


# --- conversion ----------------------------------------------------------------------


def test_proper_torsion_to_tmd_pattern_structure_and_idivf():
    torsion = _Torsion(
        "[*:1]-[*:2]-[*:3]-[*:4]", [(10.0, 0.0, 2.0), (4.0, 3.14, 1.0)], idivf=2.0
    )

    smirks, components = proper_torsion_to_tmd_pattern(torsion)

    assert smirks == "[*:1]-[*:2]-[*:3]-[*:4]"
    # k divided by idivf, phase preserved, periodicity coerced to float
    assert components == [[5.0, 0.0, 2.0], [2.0, 3.14, 1.0]]


def test_proper_torsion_defaults_idivf_to_one():
    torsion = _Torsion("[*:1]-[*:2]-[*:3]-[*:4]", [(8.0, 0.0, 3.0)])
    _, components = proper_torsion_to_tmd_pattern(torsion)
    assert components == [[8.0, 0.0, 3.0]]


# --- merge ---------------------------------------------------------------------------


def _base_ff():
    return {
        "AM1CCC": {"patterns": [("[#6:1]-[#6:2]", 0.123)]},
        "HarmonicBond": {"patterns": [["[#6:1]-[#6:2]", 300.0, 0.15]]},
        "ProperTorsion": {"patterns": [["[#6:1]-[#6:2]-[#6:3]-[#6:4]", [[1.0, 0.0, 2.0]]]]},
    }


def test_merge_appends_new_smirks():
    ff, added, replaced = merge_torsions_into_tmd(
        _base_ff(), [_Torsion("[#7:1]-[#7:2]-[#7:3]-[#7:4]", [(2.0, 0.0, 3.0)])]
    )

    assert added == ["[#7:1]-[#7:2]-[#7:3]-[#7:4]"]
    assert replaced == []
    assert len(ff["ProperTorsion"]["patterns"]) == 2
    assert ff["ProperTorsion"]["patterns"][-1] == [
        "[#7:1]-[#7:2]-[#7:3]-[#7:4]",
        [[2.0, 0.0, 3.0]],
    ]


def test_merge_overrides_matching_base_torsion():
    """A fit torsion with the same SMIRKS as a base torsion must overwrite it."""
    ff, added, replaced = merge_torsions_into_tmd(
        _base_ff(), [_Torsion("[#6:1]-[#6:2]-[#6:3]-[#6:4]", [(9.9, 0.0, 2.0)])]
    )

    assert added == []
    assert replaced == ["[#6:1]-[#6:2]-[#6:3]-[#6:4]"]
    assert len(ff["ProperTorsion"]["patterns"]) == 1
    # overwritten with the new fit value, not the original 1.0
    assert ff["ProperTorsion"]["patterns"][0] == [
        "[#6:1]-[#6:2]-[#6:3]-[#6:4]",
        [[9.9, 0.0, 2.0]],
    ]


def test_merge_preserves_other_handlers():
    base = _base_ff()
    ff, _, _ = merge_torsions_into_tmd(
        base, [_Torsion("[#7:1]-[#7:2]-[#7:3]-[#7:4]", [(2.0, 0.0, 3.0)])]
    )

    assert ff["AM1CCC"] == base["AM1CCC"]
    assert ff["HarmonicBond"] == base["HarmonicBond"]
    # input not mutated
    assert len(base["ProperTorsion"]["patterns"]) == 1


# --- collect fitted torsions ---------------------------------------------------------


def test_collect_fitted_torsions_ignores_untouched_and_dedupes():
    base = _ForceField(
        [
            _Torsion("A", [(1.0, 0.0, 2.0)]),
            _Torsion("B", [(1.0, 0.0, 2.0)]),
        ]
    )
    refit_one = _ForceField(
        [
            _Torsion("A", [(1.0, 0.0, 2.0)]),  # untouched -> excluded
            _Torsion("B", [(5.0, 0.0, 2.0)]),  # changed   -> included
            _Torsion("C", [(2.0, 0.0, 3.0)]),  # new       -> included
        ]
    )
    refit_two = _ForceField([_Torsion("C", [(2.0, 0.0, 3.0)])])  # dup -> ignored

    fitted = collect_fitted_torsions([refit_one, refit_two], base)

    assert [parameter.smirks for parameter in fitted] == ["B", "C"]


# --- (de)serialisation ---------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    ff = _base_ff()
    path = str(tmp_path / "ff.py")

    save_tmd_force_field(ff, path)
    assert load_tmd_force_field(path) == ff

    # tmd loads the file with ast.literal_eval; a leading comment must be tolerated.
    with open(path) as file:
        text = file.read()
    assert text.lstrip().startswith("#")
    assert ast.literal_eval(text) == ff


def test_tmd_proper_torsion_smirks():
    ff = {
        "ProperTorsion": {
            "patterns": [
                ["[#6:1]-[#6:2]-[#6:3]-[#6:4]", [[1.0, 0.0, 2.0]]],
                ["[#7:1]-[#7:2]-[#7:3]-[#7:4]", [[2.0, 0.0, 3.0]]],
            ]
        },
        "HarmonicBond": {"patterns": [["[#6:1]-[#6:2]", 300.0, 0.15]]},
    }
    assert tmd_proper_torsion_smirks(ff) == {
        "[#6:1]-[#6:2]-[#6:3]-[#6:4]",
        "[#7:1]-[#7:2]-[#7:3]-[#7:4]",
    }


def test_tmd_proper_torsion_smirks_missing_handler():
    assert tmd_proper_torsion_smirks({"HarmonicBond": {"patterns": []}}) == set()


def test_load_rejects_non_dict(tmp_path):
    path = str(tmp_path / "bad.py")
    with open(path, "w") as file:
        file.write("[1, 2, 3]\n")
    with pytest.raises(ValueError, match="dictionary literal"):
        load_tmd_force_field(path)


# --- toolkit-gated: real unit conversion --------------------------------------------


def test_real_proper_torsion_conversion_units():
    """Against a real OpenFF force field, the converted k must equal the OpenFF value in
    kJ/mol divided by idivf (matching tmd's own SMIRNOFF converter)."""
    pytest.importorskip("openff.toolkit")
    from openff.toolkit import ForceField

    handler = ForceField("openff_unconstrained-2.2.0.offxml").get_parameter_handler(
        "ProperTorsions"
    )
    parameter = handler.parameters[0]

    smirks, components = proper_torsion_to_tmd_pattern(parameter)

    assert smirks == parameter.smirks
    assert len(components) >= 1
    expected_k = parameter.k1.m_as("kilojoule / mole") / float(parameter.idivf1)
    assert components[0][0] == pytest.approx(expected_k)
    assert components[0][1] == pytest.approx(parameter.phase1.m_as("radian"))
    assert components[0][2] == float(parameter.periodicity1)
