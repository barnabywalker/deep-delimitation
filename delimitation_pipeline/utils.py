import subprocess
import warnings

class UncommittedError(Exception):
    pass

class UncommittedWarning(Warning):
    pass

def check_uncommitted(warn=False):
    subprocess.run(["git", "update-index", "--refresh"])
    changes = subprocess.check_output(["git", "diff-index", "--name-only", "HEAD", "--"]).strip().decode()

    msg = "Uncommited changes detected. Please consider committing your changes before training, for reproducibility."
    if changes != "":
        if warn:
            warnings.warn(msg, UncommittedWarning)
        else:
            raise UncommittedError(msg)

def get_commit_hash():
    return subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    