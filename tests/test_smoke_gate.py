from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.wait_for_smoke_then_production import validate_smoke_analysis


class SmokeGateTest(unittest.TestCase):
    def _write(self, root: Path, precontact: tuple[float, float], manipulation: tuple[float, float]) -> Path:
        path = root / "oracle_analysis.json"
        path.write_text(
            json.dumps(
                {
                    "rq2": {
                        "selected_lambda": 0.1,
                        "best_success_rate": 1.0,
                        "phase_distribution": [
                            {"phase": "precontact", "state_count": 3, "mean_E": precontact[0], "mean_D": precontact[1]},
                            {"phase": "manipulation", "state_count": 4, "mean_E": manipulation[0], "mean_D": manipulation[1]},
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_accepts_downward_e_and_positive_d_for_both_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate_smoke_analysis(
                self._write(Path(directory), (8.0, 2.0), (6.0, 4.0))
            )
        self.assertTrue(result["passed"])

    def test_rejects_capped_execution_or_zero_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate_smoke_analysis(
                self._write(Path(directory), (10.0, 0.0), (6.0, 4.0))
            )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
