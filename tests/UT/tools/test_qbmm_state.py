"""CPU-only checkpoint serialization and integrity regressions."""

import json

import pytest


def test_checkpoint_nonfinite_failure_roundtrip(tmp_path):
    from tests.ops.eagle_quant_batch_matmul_swiglu import qbmm_state as state

    path = tmp_path / "failure.json"
    record = dict(
        status="FAIL",
        reports=[
            dict(
                properties=[
                    (
                        "metrics",
                        dict(
                            cosine=float("nan"),
                            max_abs=float("inf"),
                            actual_at_max=-float("inf"),
                            passed=False,
                            matched_ratio=0.25,
                        ),
                    )
                ]
            )
        ],
    )
    state.save(path, record)
    loaded = state.read(path)
    metrics = loaded["reports"][0]["properties"][0][1]
    assert loaded["status"] == "FAIL"
    assert metrics == dict(cosine="nan", max_abs="inf", actual_at_max="-inf", passed=False, matched_ratio=0.25)
    assert not record["reports"][0]["properties"][0][1]["passed"]
    # Standard JSON consumers must not need the nonstandard NaN/Infinity extension.
    json.loads(path.read_text(), parse_constant=pytest.fail)
    assert not list(tmp_path.glob("*.pending"))


def test_checkpoint_finite_record_checksum_and_corruption(tmp_path):
    from tests.ops.eagle_quant_batch_matmul_swiglu import qbmm_state as state

    path = tmp_path / "pass.json"
    record = dict(status="PASS", metrics=dict(cosine=1.0, passed=True))
    state.save(path, record)
    assert state.read(path) == record
    payload = json.loads(path.read_text())
    assert payload["sha256"] == state.hashlib.sha256(state.canonical(record)).hexdigest()
    payload["value"]["status"] = "FAIL"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        state.read(path)
