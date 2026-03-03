# tests/test_integration.py
"""End-to-end integration tests for emcoil."""

import json
import subprocess
import sys
import os
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_cli(*args):
    """Run cli.py and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "cli.py", *args],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return result.returncode, result.stdout, result.stderr


class TestIntegration:
    def test_full_pipeline_air_core(self):
        """Full pipeline: air core, point evaluation, JSON output."""
        output_path = os.path.join(PROJECT_ROOT, "test_integration.json")
        try:
            code, out, err = run_cli(
                "--radius", "25", "--length", "100", "--amp-turns", "500",
                "--core", "air", "--point", "0", "0", "150",
                "--output", "test_integration.json"
            )
            assert code == 0, f"CLI failed with stderr: {err}"
            assert "|B|" in out

            with open(output_path) as f:
                data = json.load(f)
            assert data["inputs"]["mu_r"] == 1.0
            assert data["field"]["|B|"] > 0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_full_pipeline_iron_core(self):
        """Iron core should give stronger field than air at same point."""
        air_path = os.path.join(PROJECT_ROOT, "test_air.json")
        iron_path = os.path.join(PROJECT_ROOT, "test_iron.json")
        try:
            run_cli("--radius", "25", "--length", "100", "--amp-turns", "500",
                    "--core", "air", "--point", "0", "0", "150",
                    "--output", "test_air.json")
            run_cli("--radius", "25", "--length", "100", "--amp-turns", "500",
                    "--core", "soft-iron", "--point", "0", "0", "150",
                    "--output", "test_iron.json")

            with open(air_path) as f:
                air = json.load(f)
            with open(iron_path) as f:
                iron = json.load(f)

            assert iron["field"]["|B|"] > air["field"]["|B|"]
        finally:
            if os.path.exists(air_path):
                os.remove(air_path)
            if os.path.exists(iron_path):
                os.remove(iron_path)

    def test_custom_mu_r_override(self):
        """--mu-r should override --core preset."""
        output_path = os.path.join(PROJECT_ROOT, "test_override.json")
        try:
            code, out, err = run_cli(
                "--radius", "25", "--length", "100", "--amp-turns", "500",
                "--core", "air", "--mu-r", "500",
                "--point", "0", "0", "150",
                "--output", "test_override.json"
            )
            assert code == 0, f"CLI failed with stderr: {err}"

            with open(output_path) as f:
                data = json.load(f)
            assert data["inputs"]["mu_r"] == 500.0
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
