"""Pick a small stratified validation subset (symlinks) from the training task dir."""
import random, sys, tomllib, collections
from pathlib import Path
src, dst, per_family = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
by = collections.defaultdict(list)
for d in sorted(src.iterdir()):
    if (d / "task.toml").exists():
        by[tomllib.loads((d / "task.toml").read_text())["metadata"]["family"]].append(d)
rng = random.Random(0)
n = 0
for fam, ds in sorted(by.items()):
    for d in rng.sample(ds, min(per_family, len(ds))):
        (dst / d.name).symlink_to(d.resolve())  # absolute target: a relative src would dangle
        n += 1
print("val tasks", n, "families", len(by))
