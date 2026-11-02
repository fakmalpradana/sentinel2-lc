"""Classify AOI files for one quarter and keep every output, for review in QGIS.

    uv run python scripts/run_aois.py 2026 3 out/islands_2026q3 examples/islands/*.geojson
"""

import json
import sys
import time
from pathlib import Path

from s2lc import pipeline

if __name__ == "__main__":
    year, quarter, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    for f in sys.argv[4:]:
        t0 = time.time()
        dst = out / Path(f).stem
        files, summary = pipeline.run(json.loads(Path(f).read_text()), year, quarter, dst,
                                      ("geotiff", "geojson", "image"))
        (dst / "aoi.geojson").write_text(Path(f).read_text())
        (dst / "summary.json").write_text(json.dumps(summary, indent=1))
        top = sorted(summary["classes"], key=lambda c: -c["percent"])[:5]
        print(f"{Path(f).stem}: {summary['area_km2']:.0f} km2, {time.time() - t0:.0f}s, "
              f"clear {summary['clear_percent']}%, "
              + ", ".join(f"{c['name_en']} {c['percent']:.1f}%" for c in top), flush=True)
