"""A named library of saved maps.

A generated landscape was a throwaway: the only way to keep one was to
download a scenario file and upload it again next time, and the app always
opened on the same procedural mountain map whatever the operator had been
working on. Anything built by hand in the editor (settlements moved,
facilities renamed, ignitions placed) was gone at the end of the session.

The library keeps maps under names the operator chooses, and remembers
which one opens with the app. A World fully describes a map, so a saved map
is its dictionary; the files are gzipped because a 200x200 world is about
sixty thousand floats per layer and an uncompressed one runs to megabytes.

The index is a convenience, not the truth: it is rebuilt from the files on
disk whenever it goes missing or disagrees with them, so copying a map file
into the folder is enough to add it and deleting one is enough to remove it.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import os
import re
import shutil
import tempfile

from .world import World

#: where the maps live. Overridable so tests do not touch the real library.
ENV_DIR = "DISASTERAWARE_MAPS"
INDEX = "index.json"
EXT = ".map.json.gz"


def library_dir() -> str:
    d = os.environ.get(ENV_DIR)
    if not d:
        d = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "maps")
    os.makedirs(d, exist_ok=True)
    return d


def slugify(name: str) -> str:
    """A file name that survives a round trip through a file system.

    Turkish names are the normal case here, and a map called
    "Marmaris kıyısı" has to come back with its own name intact, so the
    display name is kept in the index and only the FILE name is folded.
    """
    _tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    s = str(name or "").translate(_tr)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_.")
    return (s or "map")[:60]


def _index_path() -> str:
    return os.path.join(library_dir(), INDEX)


def _read_index() -> dict:
    try:
        with open(_index_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and isinstance(d.get("maps"), dict):
            return d
    except Exception:
        pass
    return {"default": None, "maps": {}}


def _write_index(idx: dict) -> None:
    # atomic: a half-written index would lose the whole library
    d = library_dir()
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=1, ensure_ascii=False)
        shutil.move(tmp, _index_path())
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _describe(world: World) -> dict:
    """What the list needs to show without opening the file."""
    from . import terrain as _tr
    cfg = world.config
    try:
        n_set = len(_tr.settlements(world))
    except Exception:
        n_set = 0
    return dict(nx=int(cfg.nx), ny=int(cfg.ny),
                cell_m=float(cfg.cell_size_m),
                km=round(cfg.nx * cfg.cell_size_m / 1000.0, 2),
                settlements=int(n_set),
                assets=int(len(getattr(world, "assets", []) or [])),
                ignitions=int(len(getattr(world, "ignitions", []) or [])))


def _sync(idx: dict) -> dict:
    """Make the index agree with the files on disk."""
    d = library_dir()
    on_disk = {f[:-len(EXT)] for f in os.listdir(d) if f.endswith(EXT)}
    maps = {k: v for k, v in idx.get("maps", {}).items() if k in on_disk}
    for slug in on_disk - set(maps):
        # a file dropped in by hand: adopt it under its own file name
        maps[slug] = dict(name=slug, slug=slug, saved="", note="")
    idx["maps"] = maps
    if idx.get("default") not in maps:
        idx["default"] = None
    return idx


def list_maps() -> list:
    """Every saved map, newest first, each with its `default` flag."""
    idx = _sync(_read_index())
    _write_index(idx)
    out = []
    for slug, m in idx["maps"].items():
        r = dict(m)
        r["slug"] = slug
        r["default"] = (slug == idx.get("default"))
        out.append(r)
    out.sort(key=lambda r: str(r.get("saved") or ""), reverse=True)
    return out


def save_map(world: World, name: str, note: str = "",
             overwrite: bool = True) -> dict:
    """Write the map under `name`. Returns its index record."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("a map needs a name")
    slug = slugify(name)
    path = os.path.join(library_dir(), slug + EXT)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(name)
    # atomic again: an interrupted save must not destroy the previous map
    fd, tmp = tempfile.mkstemp(dir=library_dir(), suffix=".tmp")
    os.close(fd)
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(world.to_dict(), fh)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    idx = _sync(_read_index())
    rec = dict(name=name, slug=slug, note=str(note or ""),
               saved=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
               **_describe(world))
    idx["maps"][slug] = rec
    _write_index(idx)
    out = dict(rec)
    out["default"] = (idx.get("default") == slug)
    return out


def _resolve(name: str) -> str:
    """Accept either the display name or the slug."""
    idx = _sync(_read_index())
    if name in idx["maps"]:
        return name
    for slug, m in idx["maps"].items():
        if str(m.get("name", "")) == str(name):
            return slug
    s = slugify(name)
    if s in idx["maps"]:
        return s
    raise KeyError(name)


def load_map(name: str) -> World:
    slug = _resolve(name)
    with gzip.open(os.path.join(library_dir(), slug + EXT),
                   "rt", encoding="utf-8") as fh:
        return World.from_dict(json.load(fh))


def delete_map(name: str) -> bool:
    slug = _resolve(name)
    path = os.path.join(library_dir(), slug + EXT)
    if os.path.exists(path):
        os.unlink(path)
    idx = _sync(_read_index())
    idx["maps"].pop(slug, None)
    if idx.get("default") == slug:
        idx["default"] = None
    _write_index(idx)
    return True


def rename_map(name: str, new_name: str) -> dict:
    """Change the display name, keeping the file where it is."""
    slug = _resolve(name)
    idx = _sync(_read_index())
    idx["maps"][slug]["name"] = str(new_name or "").strip() or slug
    _write_index(idx)
    return dict(idx["maps"][slug], slug=slug,
                default=(idx.get("default") == slug))


def set_default(name: str | None) -> str | None:
    """Mark the map that opens with the app, or clear the mark."""
    idx = _sync(_read_index())
    idx["default"] = None if name is None else _resolve(name)
    _write_index(idx)
    return idx["default"]


def default_name() -> str | None:
    idx = _sync(_read_index())
    slug = idx.get("default")
    if not slug:
        return None
    return str(idx["maps"][slug].get("name") or slug)


def load_default() -> World | None:
    """The default map, or None when nothing is marked or it will not open.

    A library file that has gone bad must not stop the app from starting,
    so a failure here is a missing default rather than a crash.
    """
    idx = _sync(_read_index())
    slug = idx.get("default")
    if not slug:
        return None
    try:
        return load_map(slug)
    except Exception:
        return None
