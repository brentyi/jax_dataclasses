import sys

collect_ignore_glob = []
if sys.version_info.major == 3 and sys.version_info.minor == 7:
    collect_ignore_glob.append("*_ignore_py37.py")
