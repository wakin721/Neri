"""Version-locked null-parent guard for Flutter 3.44.6 Windows x64 release.

The installed engine faults in AccessibilityBridge::CreateRemoveReparentedNodesUpdate
at RVA 0x3a9fa. This changes only the 25-byte child/parent check, without changing
section sizes, stack layout, unwind metadata, exports or accessibility settings.
Equivalent source change: skip the child if !child || !child->parent().
Remove this hotfix after moving to an engine with a verified upstream fix.
"""

import argparse
import hashlib
from pathlib import Path
import struct

ORIGINAL_SHA256 = "a5c880a4218a04cfb3e58b103160334658122c5e43d3b818b3d9af5ab3e1fc0e"
RVA = 0x3A9ED
ORIGINAL = bytes.fromhex("4885c00f84e6000000488b40208b40483b46180f84d6000000")
# test rax,rax; jz skip; mov rcx,[rax+20h]; jrcxz skip;
# mov eax,[rcx+48h]; cmp eax,[rsi+18h]; jne reparent; skip: jmp continue; nop
# RCX is dead on both exits: the reparent path overwrites it before use,
# and the next loop iteration obtains RCX from the bridge's AXTree.
GUARDED = bytes.fromhex("4885c0740e488b4820e3088b41483b46187506e9d700000090")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_offset(data: bytes) -> int:
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[:2] != b"MZ" or data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("Not a PE DLL")
    if struct.unpack_from("<H", data, pe + 4)[0] != 0x8664:
        raise ValueError("Expected Windows x64 engine")
    count = struct.unpack_from("<H", data, pe + 6)[0]
    table = pe + 24 + struct.unpack_from("<H", data, pe + 20)[0]
    for index in range(count):
        section = table + index * 40
        _, start, size, offset = struct.unpack_from("<IIII", data, section + 8)
        if start <= RVA and RVA + len(ORIGINAL) <= start + size:
            return offset + RVA - start
    raise ValueError("Guard RVA is outside file-backed sections")


def patched_bytes(data: bytes) -> bytes:
    offset = patch_offset(data)
    # Recognize only our exact modified engine; never patch an unknown build.
    if data[offset:offset + len(GUARDED)] == GUARDED:
        restored = data[:offset] + ORIGINAL + data[offset + len(GUARDED):]
        if sha256(restored) == ORIGINAL_SHA256:
            return data
    if sha256(data) != ORIGINAL_SHA256:
        raise ValueError("Unsupported engine hash; review the new engine before applying this hotfix")
    if data[offset:offset + len(ORIGINAL)] != ORIGINAL:
        raise ValueError("Unexpected instructions at the guard RVA")
    result = data[:offset] + GUARDED + data[offset + len(ORIGINAL):]
    assert len(result) == len(data)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    data = patched_bytes(source.read_bytes())
    if destination == source and data == source.read_bytes():
        print(f"Already guarded: {destination} ({sha256(data)})")
        return
    temporary = destination.with_name(destination.name + ".hotfix-staging")
    try:
        temporary.write_bytes(data)
        if temporary.read_bytes() != data:
            raise IOError("Staged engine verification failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Guarded engine: {destination} ({sha256(data)})")


if __name__ == "__main__":
    main()
