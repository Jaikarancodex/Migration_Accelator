"""deploy_bundle validates before deploying, and skips the upload step on
a validation failure. Uses a fake `databricks` CLI (mocked subprocess.run)
since these don't need a live workspace.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from deploy.export import deploy_bundle, validate_bundle


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_validate_bundle_reports_success() -> None:
    with patch("deploy.export.subprocess.run", return_value=_completed(0, "Validation OK!")) as run:
        ok, output = validate_bundle("bundles/x", "https://h", "tok")
    assert ok is True
    assert "Validation OK" in output
    assert run.call_args.args[0] == ["databricks", "bundle", "validate"]


def test_validate_bundle_reports_failure() -> None:
    with patch("deploy.export.subprocess.run", return_value=_completed(1, "", "Invalid resource name")):
        ok, output = validate_bundle("bundles/x", "https://h", "tok")
    assert ok is False
    assert "Invalid resource name" in output


def test_deploy_bundle_skips_upload_when_validation_fails(tmp_path: Path) -> None:
    with patch(
        "deploy.export.subprocess.run", return_value=_completed(1, "", "Invalid resource name")
    ) as run:
        ok, output = deploy_bundle(tmp_path, "https://h", "tok")
    assert ok is False
    assert "bundle validate failed" in output
    assert "Invalid resource name" in output
    # only the validate call happened -- deploy was never attempted
    assert run.call_count == 1
    assert run.call_args.args[0] == ["databricks", "bundle", "validate"]


def test_deploy_bundle_proceeds_after_validation_succeeds(tmp_path: Path) -> None:
    responses = [
        _completed(0, "Validation OK!"),  # validate
        _completed(0, "Deployment complete!"),  # deploy
        _completed(0, "Name: my-bundle"),  # summary
    ]
    with patch("deploy.export.subprocess.run", side_effect=responses) as run:
        ok, output = deploy_bundle(tmp_path, "https://h", "tok")
    assert ok is True
    assert "Deployment complete" in output
    assert "Name: my-bundle" in output
    assert run.call_count == 3
    called_subcommands = [call.args[0][:3] for call in run.call_args_list]
    assert called_subcommands == [
        ["databricks", "bundle", "validate"],
        ["databricks", "bundle", "deploy"],
        ["databricks", "bundle", "summary"],
    ]
