import json
from pathlib import Path
import runpy


def test_nonfinite_diagnostics_remain_reportable() -> None:
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "fit_scalar_lgssm.py")
    )
    encoded = example["encode_report"](
        {
            "mcmc": {
                "basic_screen_passed": False,
                "max_rhat": float("nan"),
                "min_ess": float("inf"),
            },
            "finite": [1.003, 1000.0],
        }
    )
    report = json.loads(encoded)
    assert report["mcmc"] == {
        "basic_screen_passed": False,
        "max_rhat": None,
        "min_ess": None,
    }
    assert report["finite"] == [1.003, 1000.0]
