#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditor de uso de scripts y reports huérfanos que:
  1) Lee entry scripts desde dvc.yaml (deps/cmd)
  2) Construye el grafo de imports en scripts/
  3) Marca scripts no usados
  4) Marca "report orphans" en reports/ comparando contra:
       - outs de dvc.yaml (incluye dicts con path, persist, cache, etc.)
       - CUALQUIER ruta en params.yaml que empiece por "reports/"
"""

from __future__ import annotations
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Tuple, Dict, Set, Any

import yaml

ROOT = Path(__file__).resolve().parent
DVC_FILE = ROOT / "dvc.yaml"
PARAMS_FILE = ROOT / "params.yaml"

# ---------------------------
# Utilidades de lectura DVC
# ---------------------------

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _extract_outs_from_stage(stage_dict: dict) -> Iterable[str]:
    outs = []
    for o in _as_list(stage_dict.get("outs")):
        if isinstance(o, str):
            outs.append(o)
        elif isinstance(o, dict):
            # formatos tipo:
            # - path: reports/...
            #   cache: false
            # o bien:
            # - reports/absa/final_all:
            #     persist: true
            # (PyYAML lo carga como { 'reports/absa/final_all': {persist: true} })
            if "path" in o:
                outs.append(o["path"])
            else:
                # clave puede ser la ruta y el valor un dict de opciones
                for k in o.keys():
                    if isinstance(k, str):
                        outs.append(k)
    return outs

def load_dvc_entry_scripts(dvc_path: Path) -> Tuple[Set[Tuple[str,str]], Set[Tuple[str,str]], Set[str], Dict[str,Any]]:
    data = yaml.safe_load(dvc_path.read_text(encoding="utf-8"))
    stages = data.get("stages", {}) or {}
    dep_scripts = set()
    cmd_scripts = set()
    outs: Set[str] = set()

    for name, st in stages.items():
        # deps: cogemos python files bajo repo
        for d in _as_list(st.get("deps")):
            if isinstance(d, str) and d.endswith(".py"):
                dep_scripts.add((name, d))
        # cmd: heurística para extraer primer script .py invocado
        cmd = st.get("cmd") or ""
        for tok in re.split(r"\s+|&&|\|\|", str(cmd)):
            if tok.endswith(".py") and ("/" in tok or tok.startswith("scripts")):
                cmd_scripts.add((name, tok))
        # outs
        for o in _extract_outs_from_stage(st):
            outs.add(o)

    return dep_scripts, cmd_scripts, outs, stages

# ---------------------------
# Utilidades de lectura PARAMS
# ---------------------------

def _walk_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_values(v)
    else:
        yield obj

def load_params_report_like_paths(params_path: Path) -> Set[str]:
    """
    Recorre TODO params.yaml y devuelve cualquier valor string
    que empiece por 'reports/' (outdirs/archivos).
    """
    if not params_path.exists():
        return set()
    p = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    reportish: Set[str] = set()
    for v in _walk_values(p):
        if isinstance(v, str) and v.startswith("reports/"):
            reportish.add(v)
    return reportish

# ---------------------------
# Scan de scripts / imports
# ---------------------------

def scan_scripts(root: Path) -> Iterable[Path]:
    for p in (root / "scripts").rglob("*.py"):
        yield p

IMPORT_RE = re.compile(r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))', re.MULTILINE)

def module_name_for_script(path: Path, base: Path) -> str:
    rel = path.relative_to(base)
    rel_no_ext = rel.with_suffix("")
    return ".".join(rel_no_ext.parts)

def build_import_graph(root: Path):
    base = root
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
                target = target.split(" as ")[0].strip()
                graph[mod].add(target)
    return graph, path_by_module

def reachable_from_entries(entries_modules: Set[str], graph: Dict[str, Set[str]]) -> Set[str]:
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

def modules_from_paths(paths: Iterable[str], base: Path) -> Set[str]:
    mods = set()
    for p in paths:
        pth = base / p
        if pth.suffix == ".py":
            mods.add(module_name_for_script(pth, base))
    return mods

# ---------------------------
# Orphans en reports/
# ---------------------------

def report_orphans(root: Path, covered_prefixes: Iterable[str]):
    reports_root = root / "reports"
    if not reports_root.exists():
        return []
    out_prefixes = [Path(o) for o in covered_prefixes if str(o).startswith("reports/")]
    orphans = []
    for p in reports_root.rglob("*"):
        rel = p.relative_to(root)
        covered = False
        for op in out_prefixes:
            try:
                rel.relative_to(op)
                covered = True
                break
            except Exception:
                pass
        if not covered:
            orphans.append(str(rel))
    return orphans

# ---------------------------
# Main
# ---------------------------

def main():
    dep_scripts, cmd_scripts, outs_from_dvc, stages = load_dvc_entry_scripts(DVC_FILE)
    entry_script_paths = sorted({p for _, p in dep_scripts} | {p for _, p in cmd_scripts})
    print(f"[INFO] DVC stages: {len(stages)}")
    print(f"[INFO] Entry scripts from DVC: {len(entry_script_paths)}")

    # Params: sumar todo lo que empiece por reports/
    params_report_paths = load_params_report_like_paths(PARAMS_FILE)

    covered_reports = set(outs_from_dvc) | set(params_report_paths)

    graph, path_by_module = build_import_graph(ROOT)
    entry_modules = modules_from_paths(entry_script_paths, ROOT)
    reachable = reachable_from_entries(entry_modules, graph)

    used_paths = set()
    for mod in reachable | entry_modules:
        p = path_by_module.get(mod)
        if p:
            used_paths.add(str(p.relative_to(ROOT)))

    all_scripts = sorted(str(p.relative_to(ROOT)) for p in scan_scripts(ROOT))
    unused = sorted(set(all_scripts) - used_paths)

    orphans = report_orphans(ROOT, covered_reports)

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
        "report_orphans_count": len(orphans),
        "covered_reports_from_dvc": sorted(o for o in outs_from_dvc if str(o).startswith("reports/")),
        "covered_reports_from_params": sorted(o for o in params_report_paths),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote results in {out_dir}/")

if __name__ == "__main__":
    main()

