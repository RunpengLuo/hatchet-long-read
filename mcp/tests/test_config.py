"""Configuration persistence, licence classification and solver choice."""

import pytest

from hatchet_mcp.config import (
    Config,
    choose_solver,
    config_path,
    inspect_gurobi_license,
    load_config,
    reference_dir,
    save_config,
)


def _licence(tmp_path, keys):
    path = tmp_path / "gurobi.lic"
    path.write_text("\n".join(f"{k}=secret-{k.lower()}" for k in keys) + "\n")
    return str(path)


@pytest.mark.parametrize(
    "keys, kind, usable",
    [
        (["WLSACCESSID", "WLSSECRET", "LICENSEID"], "wls", True),
        (["TOKENSERVER"], "token-server", True),
        (["HOSTID", "HOSTNAME", "KEY", "TYPE"], "node-locked", False),
        (["SOMETHING"], "unknown", False),
    ],
)
def test_licence_type_from_key_names(tmp_path, keys, kind, usable):
    info = inspect_gurobi_license(_licence(tmp_path, keys))
    assert info["type"] == kind
    assert info["usable_in_container"] is usable


def test_licence_values_never_returned(tmp_path):
    info = inspect_gurobi_license(_licence(tmp_path, ["HOSTID", "KEY"]))
    assert "secret" not in repr(info)


def test_missing_and_absent_licence(tmp_path):
    assert inspect_gurobi_license(None)["type"] == "none"
    assert inspect_gurobi_license(str(tmp_path / "nope.lic"))["type"] == "missing"


def test_node_locked_licence_only_usable_locally(tmp_path):
    info = inspect_gurobi_license(_licence(tmp_path, ["HOSTID"]))
    assert choose_solver(info, "docker") == "cbc"
    assert choose_solver(info, "local") == "gurobi"


def test_config_round_trip():
    config = Config(
        runtime="docker", gurobi_license="/x/gurobi.lic", solver="gurobi", configured=True
    )
    save_config(config)
    assert config_path().exists()
    loaded = load_config()
    assert loaded.gurobi_license == "/x/gurobi.lic" and loaded.configured


def test_reference_dir_is_hatchet_repo():
    assert reference_dir(Config(runtime="docker")) == "/opt/hatchet/data/reference"
    local = Config(runtime="local", hatchet_local_source="/src/hatchet")
    assert reference_dir(local) == "/src/hatchet/data/reference"
    with pytest.raises(ValueError):
        reference_dir(Config(runtime="local"))
