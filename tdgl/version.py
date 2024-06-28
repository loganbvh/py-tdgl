__version_info__ = (0, 8, 3)
__version__ = ".".join(map(str, __version_info__))


def git_version():
    import os.path
    import subprocess

    git_hash = git_date = None

    dirname = os.path.dirname(__file__)

    if not os.path.exists(os.path.join(dirname, os.pardir, ".git")):
        return git_hash, git_date

    try:
        p = subprocess.Popen(
            ["git", "log", "-1", '--format="%H %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=dirname,
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode("utf-8").strip().replace('"', "").split("T")[0].split()
            )

    return git_hash, git_date


git_hash, git_date = git_version()

__git_revision__ = None
if git_hash is not None:
    __git_revision__ = f"{git_hash[:7]} [{git_date}]"
