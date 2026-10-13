"""Generate a learner notebook (stubs) from its reference solution.

    python scripts/stubify.py 02-linear-algebra/03-solving-linear-systems [more topic dirs...]
    python scripts/stubify.py 02-linear-algebra/capstone

Reads `solutions/exercise.py` (or `solutions/capstone.py` in a capstone directory) and writes
`exercise.py` (or `capstone.py`) next to `solutions/`. Every top-level function, and every
method of a top-level class, whose docstring starts with "stage N:" has its body replaced by
`raise NotImplementedError("stageN: implement <name>")`, keeping the signature and docstring.
" (reference solution)" is removed from the title.

One function can span two stages via a flag in its docstring's first line:
    "stage 1 (modified=False) and stage 2 (modified=True): ..."
becomes a stub that raises the right stage's message depending on the flag's value.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

STAGE = re.compile(r"^\s*stage (\d+)\b")
MULTI = re.compile(r"stage (\d+) \((\w+)=(\w+)\)")


def _stub_ranges(src: str) -> list[tuple[int, int, str, str]]:
    """(first body line, last line, indent, replacement) for every stage-tagged function."""
    out = []

    def visit(fn: ast.FunctionDef, prefix: str = "") -> None:
        doc = ast.get_docstring(fn)
        if not doc or not STAGE.match(doc) or len(fn.body) < 2:
            return
        first = fn.body[1]
        indent = " " * first.col_offset
        name = f"{prefix}{fn.name}"
        multi = MULTI.findall(doc.split("\n")[0])
        if len(multi) >= 2:
            lines = []
            for stage, flag, val in multi[1:]:
                cond = flag if val == "True" else f"not {flag}"
                lines.append(f"if {cond}:\n{indent}    raise NotImplementedError("
                             f'"stage{stage}: implement {name} with {flag}={val}")')
            s0, f0, v0 = multi[0]
            lines.append(f'raise NotImplementedError("stage{s0}: implement {name} with {f0}={v0}")')
            out.append((first.lineno, fn.end_lineno, indent, f"\n{indent}".join(lines)))
        else:
            stage = STAGE.match(doc).group(1)
            out.append((first.lineno, fn.end_lineno, indent,
                        f'raise NotImplementedError("stage{stage}: implement {name}")'))

    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            visit(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    visit(sub, prefix=f"{node.name}.")
    return out


def stubify(src: str) -> str:
    lines = src.splitlines(keepends=True)
    for start, end, indent, replacement in sorted(_stub_ranges(src), reverse=True):
        lines[start - 1:end] = [f"{indent}{replacement}\n"]
    return "".join(lines).replace(" (reference solution)", "")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for topic in map(Path, argv):
        name = "capstone.py" if topic.name == "capstone" else "exercise.py"
        src = topic / "solutions" / name
        dst = topic / name
        dst.write_text(stubify(src.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
