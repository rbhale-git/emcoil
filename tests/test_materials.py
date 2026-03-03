# tests/test_materials.py
import pytest
from emcoil.materials import get_mu_r, MATERIAL_PRESETS


def test_presets_exist():
    assert "air" in MATERIAL_PRESETS
    assert "water" in MATERIAL_PRESETS
    assert "soft-iron" in MATERIAL_PRESETS
    assert "silicon-steel" in MATERIAL_PRESETS
    assert "ferrite" in MATERIAL_PRESETS
    assert "mu-metal" in MATERIAL_PRESETS


def test_air_mu_r():
    assert get_mu_r("air") == 1.0


def test_water_mu_r():
    assert get_mu_r("water") == pytest.approx(0.999992)


def test_soft_iron_mu_r():
    assert get_mu_r("soft-iron") == 800.0


def test_custom_mu_r_overrides_preset():
    result = get_mu_r("soft-iron", custom_mu_r=1200.0)
    assert result == 1200.0


def test_custom_mu_r_without_preset():
    result = get_mu_r(None, custom_mu_r=42.0)
    assert result == 42.0


def test_no_material_no_custom_defaults_to_air():
    result = get_mu_r(None)
    assert result == 1.0


def test_unknown_material_raises():
    with pytest.raises(ValueError, match="Unknown material"):
        get_mu_r("unobtainium")


def test_custom_mu_r_must_be_positive():
    with pytest.raises(ValueError, match="mu_r must be positive"):
        get_mu_r(None, custom_mu_r=-1.0)
