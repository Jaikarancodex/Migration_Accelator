"""Git operations for versioning migrated bundles from the app.

Lets the review app configure which repository/branch the generated Asset
Bundles are committed to, then commit and push them as part of deploy — so
every deployed workflow is also versioned in git (the "workflow goes into
git based on DAB, then deploys to Databricks" flow).

Auth: pushes use whatever credentials git is already configured with. An
optional GitHub token can be supplied for a single push; it is used to
build an authenticated URL in-memory only and is never written to git
config, the remote, or disk.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(repo_dir: str | Path, args: list[str], timeout: int = 120) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise GitError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def repo_info(repo_dir: str | Path) -> dict[str, str | bool]:
    """Current branch and origin URL; is_repo=False if not a git repo."""
    try:
        branch = _git(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    except GitError:
        return {"is_repo": False, "branch": "", "remote_url": ""}
    try:
        url = _git(repo_dir, ["remote", "get-url", "origin"])
    except GitError:
        url = ""
    return {"is_repo": True, "branch": branch, "remote_url": _redact(url)}


def _redact(url: str) -> str:
    """Hide any credentials embedded in a remote URL before display."""
    return re.sub(r"//[^@/]+@", "//***@", url)


def set_remote(repo_dir: str | Path, url: str, remote: str = "origin") -> None:
    """Point `remote` at `url`, adding it if it doesn't exist yet."""
    try:
        _git(repo_dir, ["remote", "get-url", remote])
        _git(repo_dir, ["remote", "set-url", remote, url])
    except GitError:
        _git(repo_dir, ["remote", "add", remote, url])


def _authenticated_url(url: str, token: str) -> str:
    """Inject a GitHub token into an https URL for a one-shot push."""
    if url.startswith("https://") and "@" not in url.split("//", 1)[1].split("/", 1)[0]:
        return url.replace("https://", f"https://x-access-token:{token}@", 1)
    return url


def commit_and_push(
    repo_dir: str | Path,
    paths: list[str],
    message: str,
    branch: str,
    remote: str = "origin",
    token: str | None = None,
) -> tuple[bool, str]:
    """Stage `paths`, commit, and push to `remote`/`branch`.

    Returns (ok, log). A no-op commit (nothing staged changed) is reported as
    success with a note rather than an error.
    """
    log: list[str] = []
    try:
        _git(repo_dir, ["add", *paths])
        status = _git(repo_dir, ["status", "--porcelain", *paths])
        if not status:
            log.append("Nothing to commit for the given paths (already up to date).")
        else:
            _git(repo_dir, ["commit", "-m", message])
            log.append(f"Committed: {message.splitlines()[0]}")

        push_target = remote
        if token:
            url = _git(repo_dir, ["remote", "get-url", remote])
            push_target = _authenticated_url(url, token)
        # HEAD:branch pushes the current commit to the named branch.
        push_out = _git(repo_dir, ["push", push_target, f"HEAD:{branch}"], timeout=300)
        log.append(_redact(push_out) or f"Pushed to {remote}/{branch}.")
        return True, "\n".join(log)
    except GitError as exc:
        log.append(f"Git error: {_redact(str(exc))}")
        return False, "\n".join(log)
