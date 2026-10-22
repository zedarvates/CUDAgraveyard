"""Validate source and template structure without importing GPU/API dependencies."""

import argparse
import ast
import json
from pathlib import Path


def validate(root):
    errors = []
    sources = sorted([root / "launch_demon.py", *root.glob("utils/*.py"), *root.glob("scripts/*.py")])
    templates = sorted(root.glob("templates/*.json"))
    for path in sources:
        try:
            source = path.read_text(encoding="utf-8")
            if source.startswith("cat >") or source.rstrip().endswith("\nEOF"):
                raise ValueError("shell here-document wrapper in a Python file")
            ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    if not templates:
        errors.append("templates/: no JSON templates found")
    for path in templates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("template must be an object")
            if not isinstance(data.get("task"), str) or not data["task"].strip():
                raise ValueError("task must be a non-empty string")
            for name in ("metrics", "constraints"):
                if not isinstance(data.get(name), list) or not data[name]:
                    raise ValueError(f"{name} must be a non-empty list")
        except (OSError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: {error}")
    return {"python_files": len(sources), "templates": len(templates), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    result = validate(parser.parse_args().root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
