from pathlib import Path
from textwrap import dedent

hints_message = dedent("""\
<div id="hintMessage" style="position: relative; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9;">
    <button onclick="document.getElementById('hintMessage').style.display = 'none'" style="position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 18px; cursor: pointer;">&times;</button>
    <div style="font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #31708f;">
        Hints Message
    </div>
    <div style="padding: 10px; border: 1px solid #31708f; border-radius: 5px; background-color: #f9f9f9;">
        Hints: Enterprise function is not supported on your system.
    </div>
    <p style="margin-top: 15px;">
        If you need Enterprise Level Support for your system or business, please send an email to
        <a href="mailto:business@siliconflow.com" style="color: #31708f; text-decoration: none;">business@siliconflow.com</a>.
        <br>
        Tell us about your use case, deployment scale, and requirements.
    </p>
    <p>
        <strong>GitHub Issue:</strong>
        <a href="https://github.com/siliconflow/onediff/issues" style="color: #31708f; text-decoration: none;">https://github.com/siliconflow/onediff/issues</a>
    </p>
</div>
""")

all_compiler_caches = []


def all_compiler_caches_path():
    import modules.shared as shared

    caches_path = Path(shared.opts.onediff_compiler_caches_path)
    if not caches_path.exists():
        caches_path.mkdir(parents=True)
    return shared.opts.onediff_compiler_caches_path


def get_all_compiler_caches():
    global all_compiler_caches
    if len(all_compiler_caches) == 0:
        refresh_all_compiler_caches()
    return all_compiler_caches


def refresh_all_compiler_caches(path: Path = None):
    global all_compiler_caches
    path = path or all_compiler_caches_path()
    all_compiler_caches = [f.stem for f in Path(path).iterdir() if f.is_file()]
