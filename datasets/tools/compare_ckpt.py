"""Compare two torch .pt state_dicts WITHOUT torch installed.

A torch.save file is a zip: `archive/data.pkl` holds the pickled state_dict, and each
tensor's bytes live in `archive/data/<storage_key>`. The pickle references storages via
persistent ids, so a small custom Unpickler can map tensor name -> zip member, after which
comparing member CRC32 tells us whether that tensor's bytes changed between checkpoints.

Usage: python compare_ckpt.py <old.pt> <new.pt> [substring ...]
"""

from __future__ import annotations

import pickle
import sys
import zipfile


class _StorageRef:
  __slots__ = ("key",)

  def __init__(self, key: str) -> None:
    self.key = key

  def __repr__(self) -> str:  # pragma: no cover
    return f"<storage {self.key}>"


class _Unpickler(pickle.Unpickler):
  def persistent_load(self, pid):  # noqa: D102
    # ('storage', storage_type, key, location, numel)
    return _StorageRef(str(pid[2]))

  def find_class(self, module, name):  # noqa: D102
    # The state_dict pickle needs these rebuild helpers, which we never call: we only
    # walk the graph for storage references.
    if module.startswith("torch"):
      class _Stub:
        def __init__(self, *a, **k):
          self.__args = a

        def __repr__(self):
          return "<rebuild>"

      return _Stub
    return super().find_class(module, name)


def _prefix(zf: zipfile.ZipFile) -> str:
  """torch.save puts everything under a folder named after the file stem."""
  for n in zf.namelist():
    if n.endswith("/data.pkl"):
      return n[: -len("data.pkl")]
  raise SystemExit("no */data.pkl member found")


def storage_keys(path: str) -> dict[str, str]:
  """Map flattened state_dict key -> zip member name of its storage."""
  out: dict[str, str] = {}
  with zipfile.ZipFile(path) as zf:
    with zf.open(_prefix(zf) + "data.pkl") as fh:
      obj = _Unpickler(fh).load()

  def walk(node, prefix: str):
    if isinstance(node, _StorageRef):
      out.setdefault(prefix, node.key)
      return
    if isinstance(node, dict):
      for k, v in node.items():
        walk(v, f"{prefix}.{k}" if prefix else str(k))
      return
    if isinstance(node, (list, tuple)):
      for i, v in enumerate(node):
        walk(v, f"{prefix}[{i}]")
      return
    # torch rebuild stubs keep their args
    args = getattr(node, "_Stub__args", None) or getattr(node, "__args", None)
    if args:
      for i, v in enumerate(args):
        walk(v, f"{prefix}<{i}>")

  walk(obj, "")
  return out


def member_crc(path: str, key: str) -> int | None:
  with zipfile.ZipFile(path) as zf:
    try:
      return zf.getinfo(f"{_prefix(zf)}data/{key}").CRC
    except KeyError:
      return None


def main() -> int:
  if len(sys.argv) < 3:
    print(__doc__)
    return 2
  old, new = sys.argv[1], sys.argv[2]
  needles = sys.argv[3:] or ["encoders.velocity", "encoders.motion"]

  a, b = storage_keys(old), storage_keys(new)
  print(f"tensors: old={len(a)}  new={len(b)}")
  common = sorted(set(a) & set(b))
  print(f"common tensors: {len(common)}")

  changed, same = [], 0
  for k in common:
    ca, cb = member_crc(old, a[k]), member_crc(new, b[k])
    if ca is None or cb is None:
      continue
    if ca != cb:
      changed.append(k)
    else:
      same += 1
  print(f"changed tensors: {len(changed)}   identical: {same}")

  for needle in needles:
    hits = [k for k in common if needle in k]
    hits_changed = [k for k in hits if k in changed]
    print(f"\n[{needle}] {len(hits)} tensors, {len(hits_changed)} changed")
    for k in hits[:8]:
      print(f"  {'CHANGED' if k in changed else 'same   '}  {k}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())