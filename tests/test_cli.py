# tests/test_cli.py
import json
import os
import subprocess
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_cli(*args):
    """Run cli.py and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "cli.py", *args],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return result.returncode, result.stdout, result.stderr


class TestCLI:
    def test_point_evaluation_runs(self):
        """Basic point evaluation should exit 0."""
        output_path = os.path.join(PROJECT_ROOT, "test_eval_results.json")
        try:
            code, out, err = run_cli(
                "--radius", "25", "--length", "100", "--amp-turns", "500",
                "--core", "air", "--point", "0", "0", "150",
                "--output", "test_eval_results.json"
            )
            assert code == 0
            assert "Bz" in out
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_custom_mu_r(self):
        """--mu-r should override --core."""
        output_path = os.path.join(PROJECT_ROOT, "test_mur_results.json")
        try:
            code, out, err = run_cli(
                "--radius", "25", "--length", "100", "--amp-turns", "500",
                "--mu-r", "1200", "--point", "0", "0", "150",
                "--output", "test_mur_results.json"
            )
            assert code == 0
            assert "mu_r: 1200" in out
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_json_output(self):
        """Should write results.json with field data."""
        output_path = os.path.join(PROJECT_ROOT, "test_results.json")
        try:
            code, out, err = run_cli(
                "--radius", "25", "--length", "100", "--amp-turns", "500",
                "--core", "soft-iron", "--point", "0", "0", "150",
                "--output", "test_results.json"
            )
            assert code == 0
            with open(output_path) as f:
                data = json.load(f)
            assert "Bz" in data["field"]
            assert "|B|" in data["field"]
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_missing_required_args(self):
        """Should fail if coil params are missing."""
        code, out, err = run_cli("--point", "0", "0", "150")
        assert code != 0
