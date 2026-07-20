import subprocess
from pathlib import Path

import pytest

from deploy.gitops import GitError, commit_and_push, repo_info, set_remote


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)


def test_repo_info_reports_branch_and_no_remote(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    info = repo_info(tmp_path)
    assert info["is_repo"] is True
    assert info["branch"] == "main"
    assert info["remote_url"] == ""


def test_repo_info_on_non_repo(tmp_path: Path) -> None:
    assert repo_info(tmp_path)["is_repo"] is False


def test_set_remote_adds_then_updates_and_redacts_credentials(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    set_remote(tmp_path, "https://token@github.com/you/repo.git")
    assert repo_info(tmp_path)["remote_url"] == "https://***@github.com/you/repo.git"
    set_remote(tmp_path, "https://github.com/you/other.git")
    assert repo_info(tmp_path)["remote_url"] == "https://github.com/you/other.git"


def test_commit_and_push_to_local_bare_remote(tmp_path: Path) -> None:
    bare = tmp_path / "remote.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "-b", "main"], cwd=bare, check=True, capture_output=True)

    work = tmp_path / "work"
    work.mkdir()
    _init_repo(work)
    set_remote(work, str(bare))

    (work / "bundle").mkdir()
    (work / "bundle" / "databricks.yml").write_text("bundle: {}", encoding="utf-8")

    ok, log = commit_and_push(work, ["bundle"], "Add bundle", "main")
    assert ok, log
    # the commit landed on the bare remote
    out = subprocess.run(
        ["git", "log", "--oneline"], cwd=bare, capture_output=True, text=True, check=True
    )
    assert "Add bundle" in out.stdout


def test_commit_and_push_noop_when_nothing_changed(tmp_path: Path) -> None:
    bare = tmp_path / "remote.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", "-b", "main"], cwd=bare, check=True, capture_output=True)
    work = tmp_path / "work"
    work.mkdir()
    _init_repo(work)
    set_remote(work, str(bare))

    ok, log = commit_and_push(work, ["README.md"], "no change", "main")
    assert ok
    assert "Nothing to commit" in log


def test_commit_and_push_reports_failure_on_bad_remote(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    _init_repo(work)
    set_remote(work, str(tmp_path / "does-not-exist.git"))
    (work / "f.txt").write_text("x", encoding="utf-8")
    ok, log = commit_and_push(work, ["f.txt"], "msg", "main")
    assert ok is False
    assert "Git error" in log


def test_git_error_raised_for_missing_binary_path(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        set_remote(tmp_path, "https://github.com/x/y.git")  # not a repo -> git error
