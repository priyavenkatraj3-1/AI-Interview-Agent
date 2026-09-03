"""
Proves hidden coding-round test cases' inputs and expected outputs are
never sent to the student-facing /submit response, while the non-scored
/run endpoint (public/sample tests only) is unaffected.

Exercised through the real FastAPI TestClient + real local executor (same
setup as test_coding_api.py) so this is a genuine end-to-end proof of the
response the frontend actually receives, not just a unit test of
_public_results in isolation. See app.services.coding_service._public_results
and its reveal_io flag.
"""
from app.services.coding_service import _HIDDEN_IO_PLACEHOLDER

CORRECT_CODE = "def add_two(a, b):\n    return a + b\n"


def test_submit_response_never_exposes_hidden_test_inputs_or_expected_outputs(client, session_id):
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": CORRECT_CODE})
    assert response.status_code == 200
    body = response.json()

    assert body["results"], "expected at least one hidden test result"
    for result in body["results"]:
        assert result["input"] == _HIDDEN_IO_PLACEHOLDER
        assert result["expected_output"] == _HIDDEN_IO_PLACEHOLDER

    # Grading/correctness is still fully communicated -- only the raw
    # hidden test args/expected values are withheld.
    assert body["is_correct"] is True
    assert body["passed_count"] == body["total_count"] == 3


def test_submit_response_hides_hidden_io_even_for_a_failing_submission(client, session_id):
    wrong_code = "def add_two(a, b):\n    return a - b\n"
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = client.post(f"/api/sessions/{session_id}/coding/submit", json={"code": wrong_code})
    assert response.status_code == 200
    body = response.json()

    assert body["is_correct"] is False
    for result in body["results"]:
        assert result["input"] == _HIDDEN_IO_PLACEHOLDER
        assert result["expected_output"] == _HIDDEN_IO_PLACEHOLDER
        # actual_output (the candidate's own function's return value) is
        # still shown -- only the hidden test's own args/expected are hidden.
        if result["error"] is None:
            assert result["actual_output"] is not None


def test_run_response_still_shows_real_public_sample_test_io(client, session_id):
    # Regression check: only hidden (submit) results are redacted -- the
    # non-scored /run endpoint against public/sample tests must be unaffected.
    client.post(f"/api/sessions/{session_id}/coding/start")
    response = client.post(f"/api/sessions/{session_id}/coding/run", json={"code": CORRECT_CODE})
    assert response.status_code == 200
    body = response.json()

    result = body["results"][0]
    assert result["input"] == "[2, 3]"
    assert result["expected_output"] == "5"
    assert result["input"] != _HIDDEN_IO_PLACEHOLDER
