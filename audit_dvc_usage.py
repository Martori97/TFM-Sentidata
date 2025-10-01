#!/usr/bin/env python3
import os, re, sys, yaml, json
from pathlib import Path
from collections import defaultdict, deque

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
DVC_FILE = ROOT / "dvc.yaml"

def load_dvc_entry_scripts(dvc_path: Path):
    if not dvc_path.exists():
        print(f"[ERROR] dvc.yaml not found at {dvc_path}")
        sys.exit(1)
    dvc = yaml.safe_load(dvc_path.read_text(encoding="utf-8"))
    stages = dvc.get("stages", {})
    dep_scripts, cmd_scripts, outs = set(), set(), set()
    for name, spec in stages.items():
        cmd = spec.get("cmd", "") or ""
        deps = spec.get("deps", []) or []
        # deps puede ser lista de strings o dicts
        for d in deps:
            if isinstance(d, dict):
                dep = next(iter(d.keys()))
            else:
                dep = d
            if isinstance(dep, str) and dep.startswith("scripts/") and dep.endswith((".py",".sh")):
                dep_scripts.add((name, str(dep)))
        # scripts en cmd
        for m in re.findall(r"(scripts/[^\s;|&]+?\.py)", cmd):
            cmd_scripts.add((name, m))
        # outs
        outs_list = spec.get("outs", []) or []
        for o in outs_list:
            if isinstance(o, dict):
                out = next(iter(o.keys()))
            else:
                out = o
            outs.add(str(out))
    return dep_scripts, cmd_scripts, outs, stages

def scan_scripts(root: Path):
    scripts = []
    for p in (root / "scripts").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        scripts.append(p)
    return scripts

IMPORT_RE = re.compile(r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))', re.MULTILINE)

def module_name_for_script(path: Path, base: Path):
    rel = path.relative_to(base)
    rel_no_ext = rel.with_suffix("")
    return ".".join(rel_no_ext.parts)

def build_import_graph(root: Path):
    base = root
    scripts_dir = base / "scripts"
    graph = defaultdict(set)
    path_by_module = {}
    for py in scan_scripts(base):
        mod = module_name_for_script(py, base)
        path_by_module[mod] = py
    for mod, py in path_by_module.items():
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in IMPORT_RE.finditer(text):
            a, b = m.groups()
            target = (a or b) or ""
            if target.startswith("scripts."):
                target_base = target.split(" as ")[0].strip()
                graph[mod].add(target_base)
    return graph, path_by_module

def reachable_from_entries(entries_modules, graph):
    seen = set()
    dq = deque(entries_modules)
    while dq:
        u = dq.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in graph.get(u, ()):
            if v not in seen:
                dq.append(v)
    return seen

def modules_from_paths(paths, base):
    mods = set()
    for p in paths:
        pth = base / p
        if pth.suffix == ".py":
            mods.add(module_name_for_script(pth, base))
    return mods

def report_orphans(root: Path, outs_set):
    reports_root = root / "reports"
    if not reports_root.exists():
        return []
    out_prefixes = [Path(o) for o in outs_set if str(o).startswith("reports/")]
    orphans = []
    for p in reports_root.rglob("*"):
        rel = p.relative_to(root)
        rel_str = str(rel)
        covered = False
        for op in out_prefixes:
            try:
                rel.relative_to(op)
                covered = True
                break
            except Exception:
                pass
        if not covered:
            orphans.append(rel_str)
    return orphans

def main():
    dep_scripts, cmd_scripts, outs, stages = load_dvc_entry_scripts(DVC_FILE)
    entry_script_paths = sorted({p for _, p in dep_scripts} | {p for _, p in cmd_scripts})
    print(f"[INFO] DVC stages: {len(stages)}")
    print(f"[INFO] Entry scripts from DVC: {len(entry_script_paths)}")

    graph, path_by_module = build_import_graph(ROOT)
    entry_modules = modules_from_paths(entry_script_paths, ROOT)
    reachable = reachable_from_entries(entry_modules, graph)

    used_paths = set()
    for mod in reachable | entry_modules:
        p = path_by_module.get(mod)
        if p:
            used_paths.add(str(p.relative_to(ROOT)))

    all_scripts = sorted([str(p.relative_to(ROOT)) for p in scan_scripts(ROOT)])
    unused = sorted(set(all_scripts) - used_paths)
    orphans = report_orphans(ROOT, outs)

    out_dir = ROOT / "audit_out"
    out_dir.mkdir(exist_ok=True)

    (out_dir / "entry_scripts.csv").write_text("\n".join(entry_script_paths), encoding="utf-8")
    (out_dir / "used_scripts.csv").write_text("\n".join(sorted(used_paths)), encoding="utf-8")
    (out_dir / "unused_scripts.csv").write_text("\n".join(unused), encoding="utf-8")
    (out_dir / "report_orphans.csv").write_text("\n".join(orphans), encoding="utf-8")
    summary = {
        "stages": len(stages),
        "entry_scripts_count": len(entry_script_paths),
        "all_scripts_count": len(all_scripts),
        "used_scripts_count": len(used_paths),
        "unused_scripts_count": len(unused),
        "report_orphans_count": len(orphans)
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote results in {out_dir}/")

if __name__ == "__main__":
    main()
