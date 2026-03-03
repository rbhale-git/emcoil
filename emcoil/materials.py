# emcoil/materials.py

MATERIAL_PRESETS = {
    "air": 1.0,
    "water": 0.999992,
    "soft-iron": 800.0,
    "silicon-steel": 4000.0,
    "ferrite": 1000.0,
    "mu-metal": 20000.0,
}


def get_mu_r(material_name=None, custom_mu_r=None):
    """Resolve relative permeability from material name and/or custom value.

    custom_mu_r overrides the preset if both are provided.
    If neither is provided, defaults to air (mu_r=1).
    """
    if custom_mu_r is not None:
        if custom_mu_r <= 0:
            raise ValueError("mu_r must be positive")
        return float(custom_mu_r)

    if material_name is None:
        return 1.0

    if material_name not in MATERIAL_PRESETS:
        raise ValueError(
            f"Unknown material '{material_name}'. "
            f"Available: {', '.join(sorted(MATERIAL_PRESETS.keys()))}"
        )

    return MATERIAL_PRESETS[material_name]
