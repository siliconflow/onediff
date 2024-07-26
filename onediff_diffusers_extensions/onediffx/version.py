_base_version = "1.2.0.dev1"


def get_version():
    global _base_version
    try:
        import subprocess

        commit_id = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return f"{_base_version}+git.{commit_id}"
    except Exception as e:
        print(f"Error retrieving git commit ID: {e}")
        return _base_version


_version = get_version()
