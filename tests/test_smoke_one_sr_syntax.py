from pathlib import Path
import ast


def test_smoke_one_sr_dnn_script_has_valid_python_syntax():
    path = Path("scripts/smoke_one_sr_dnn.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
