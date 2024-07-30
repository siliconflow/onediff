try:
    import git  # type: ignore
except ImportError:
    print("GitPython library not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "GitPython"])
    import git  # type: ignore

from datetime import datetime


class FunctionSelectorByCommitDate:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)

    def _get_current_commit_date(self):
        """
        Get the date of the current commit.
        """
        current_commit = self.repo.head.commit
        current_commit_date = datetime.fromtimestamp(current_commit.committed_date)
        return current_commit, current_commit_date

    def _get_commit_date(self, commit_hash):
        """
        Get the date of a specific commit.
        """
        commit = self.repo.commit(commit_hash)
        commit_date = datetime.fromtimestamp(commit.committed_date)
        return commit_date

    def __call__(self, iterable, default_func=None):
        """
        Select a function based on the commit date.

        Args:
            iterable (iterable): An iterable of tuples, where each tuple contains
                a commit hash and a corresponding function.
            default_func (callable, optional): The default function to return
                if no match is found.

        Returns:
            callable: The selected function.
        """
        cur_commit, cur_date = self._get_current_commit_date()
        sel_date, sel_func = None, default_func
        for pair in iterable:
            commit_hash, func = pair
            other_commit_date = self._get_commit_date(commit_hash)
            if cur_date > other_commit_date and (
                sel_date and other_commit_date > sel_date
            ):
                sel_date = other_commit_date
                sel_func = func

        return sel_func
