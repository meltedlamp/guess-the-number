"""Write docs/brain.bin from the factory weights for the browser page."""

import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "weights.npz"
DST = ROOT / "docs" / "brain.bin"


def _write(handle, array, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    handle.write(struct.pack("<I", array.ndim))
    handle.write(struct.pack("<" + "I" * array.ndim, *array.shape))
    handle.write(array.tobytes())


def main():
    data = np.load(SRC)
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("wb") as handle:
        handle.write(b"REED")
        handle.write(struct.pack("<I", 1))
        for name in ("W1", "b1", "W2", "b2", "W3", "b3", "anchor_x"):
            _write(handle, data[name], "<f4")
        _write(handle, data["anchor_y"], "<i4")
    print(f"wrote {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
