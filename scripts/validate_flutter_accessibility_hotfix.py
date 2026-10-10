"""Verify the version-locked Flutter guard with emulation and native Windows execution; DLLs are read only."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dll', type=Path, required=True)
    parser.add_argument('--patched-dll', type=Path, required=True)
    parser.add_argument('--dependency-dir', type=Path, help='Optional directory with capstone, pefile and unicorn')
    parser.add_argument('--report-file', type=Path, help='Optional new report file; otherwise print only')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.report_file and args.report_file.exists():
        raise FileExistsError(args.report_file)
    import ctypes as c
    import json
    import random
    import struct

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / 'scripts'))
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    import flutter_accessibility_hotfix as fix
    import capstone
    import pefile
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UcError
    from unicorn.x86_const import UC_X86_REG_EFLAGS, UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15, UC_X86_REG_RAX, UC_X86_REG_RBP, UC_X86_REG_RBX, UC_X86_REG_RCX, UC_X86_REG_RDI, UC_X86_REG_RDX, UC_X86_REG_RIP, UC_X86_REG_RSI, UC_X86_REG_RSP

    patched_path = args.patched_dll.resolve()
    original = args.original_dll.read_bytes()
    patched = patched_path.read_bytes()
    assert fix.patched_bytes(original) == patched
    assert fix.patched_bytes(patched) == patched
    changed = [i for i, (a, b) in enumerate(zip(original, patched)) if a != b]
    offset = fix.patch_offset(original)
    assert all(offset <= i < offset + 25 for i in changed)
    for corrupt in [original[:-1] + bytes([original[-1] ^ 1]), patched[:-1] + bytes([patched[-1] ^ 1])]:
        try:
            fix.patched_bytes(corrupt)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported engine accepted')

    BASE = 0x180000000
    CHILD, PARENT, UPDATE, STACK = 0x300000000, 0x400000000, 0x500000000, 0x600000000
    START, REPARENT, SKIP = BASE + fix.RVA, BASE + 0x3aa06, BASE + 0x3aadc
    REGS = [UC_X86_REG_RAX, UC_X86_REG_RDX, UC_X86_REG_RSI, UC_X86_REG_RDI,
            UC_X86_REG_RBX, UC_X86_REG_RBP, UC_X86_REG_RSP, UC_X86_REG_R12,
            UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15, UC_X86_REG_EFLAGS]


    def machine(code):
        uc = Uc(UC_ARCH_X86, UC_MODE_64)
        uc.mem_map(BASE, 0x1600000)
        uc.mem_write(BASE, pefile.PE(data=code).get_memory_mapped_image())
        for address in [CHILD, PARENT, UPDATE, STACK]:
            uc.mem_map(address, 0x1000)
        return uc


    old, new = machine(original), machine(patched)


    def execute(uc, child_exists, parent_exists, old_id, new_id):
        uc.mem_write(CHILD + 0x20, struct.pack('<Q', PARENT if parent_exists else 0))
        uc.mem_write(PARENT + 0x48, struct.pack('<I', old_id & 0xffffffff))
        uc.mem_write(UPDATE + 0x18, struct.pack('<I', new_id & 0xffffffff))
        for i, reg in enumerate(REGS[:-1]):
            uc.reg_write(reg, 0x123400 + i)
        uc.reg_write(UC_X86_REG_RAX, CHILD if child_exists else 0)
        uc.reg_write(UC_X86_REG_RCX, 0x778899)
        uc.reg_write(UC_X86_REG_RSI, UPDATE)
        uc.reg_write(UC_X86_REG_RSP, STACK + 0x800)
        uc.reg_write(UC_X86_REG_EFLAGS, 0x202)
        # Only this guard is under test; no mocked continuation or allocator executes.
        uc.reg_write(UC_X86_REG_RIP, START)
        for _ in range(20):
            pc = uc.reg_read(UC_X86_REG_RIP)
            if pc in (REPARENT, SKIP):
                return pc, [uc.reg_read(reg) for reg in REGS]
            uc.emu_start(pc, BASE + 0x3ac54, count=1)
        raise AssertionError('Guard did not exit')


    try:
        execute(old, True, False, 0, 1)
    except UcError as exc:
        baseline_fault = dict(error=str(exc), rva=hex(old.reg_read(UC_X86_REG_RIP) - BASE))
        assert baseline_fault['rva'] == '0x3a9fa'
    else:
        raise AssertionError('Original null-parent fault was not reproduced')

    rng = random.Random(20261009)
    cases = [(False, False, 0, 1), (True, False, 0, 1)]
    for i in range(4096):
        parent_id = rng.randint(-2**31, 2**31 - 1)
        update_id = parent_id if i % 2 else rng.randint(-2**31, 2**31 - 1)
        cases.append((True, True, parent_id, update_id))
    for case in cases:
        got = execute(new, *case)
        expected = REPARENT if case[0] and case[1] and case[2] != case[3] else SKIP
        assert got[0] == expected, (case, got[0], expected)
        if not case[0] or case[1]:
            assert execute(old, *case) == got, case

    # Native execution of exactly the guarded instructions, using the Windows x64 ABI.
    # The wrapper saves RSI and replaces the two continuations with boolean returns.
    prefix = bytes.fromhex('564889d64889c8')  # push rsi; mov rsi,rdx; mov rax,rcx
    wrapper = bytearray(b'\xcc' * 0x200)
    wrapper[:len(prefix)] = prefix
    wrapper[len(prefix):len(prefix)+25] = fix.GUARDED
    for rva, result in [(0x3aa06, 1), (0x3aadc, 0)]:
        target = len(prefix) + rva - fix.RVA
        wrapper[target:target + 7] = b'\xb8' + struct.pack('<I', result) + b'\x5e\xc3'
    kernel = c.WinDLL('kernel32', use_last_error=True)
    kernel.VirtualAlloc.argtypes = [c.c_void_p, c.c_size_t, c.c_uint32, c.c_uint32]
    kernel.VirtualAlloc.restype = c.c_void_p
    kernel.VirtualProtect.argtypes = [c.c_void_p, c.c_size_t, c.c_uint32, c.POINTER(c.c_uint32)]
    kernel.VirtualFree.argtypes = [c.c_void_p, c.c_size_t, c.c_uint32]
    kernel.FlushInstructionCache.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t]
    kernel.GetCurrentProcess.restype = c.c_void_p
    address = kernel.VirtualAlloc(None, 0x1000, 0x3000, 4)
    assert address
    native_count = 0
    try:
        c.memmove(address, bytes(wrapper), len(wrapper))
        previous = c.c_uint32()
        assert kernel.VirtualProtect(address, 0x1000, 0x20, c.byref(previous))
        assert kernel.FlushInstructionCache(kernel.GetCurrentProcess(), address, len(wrapper))
        function = c.WINFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)(address)
        child, parent, update = c.create_string_buffer(256), c.create_string_buffer(256), c.create_string_buffer(256)
        for child_exists, parent_exists, parent_id, update_id in cases:
            struct.pack_into('<Q', child, 0x20, c.addressof(parent) if parent_exists else 0)
            struct.pack_into('<I', parent, 0x48, parent_id & 0xffffffff)
            struct.pack_into('<I', update, 0x18, update_id & 0xffffffff)
            result = function(c.addressof(child) if child_exists else 0, c.addressof(update))
            assert result == int(child_exists and parent_exists and parent_id != update_id)
            native_count += 1
    finally:
        assert kernel.VirtualFree(address, 0, 0x8000)

    # Also exercise the Windows loader against the staged, otherwise complete DLL.
    library = c.WinDLL(str(patched_path))
    assert library.FlutterDesktopEngineCreate
    disassembler = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    instructions = [f'{x.address:x}: {x.mnemonic} {x.op_str}' for x in disassembler.disasm(fix.GUARDED, fix.RVA)]
    result = dict(original_sha256=fix.sha256(original), patched_sha256=fix.sha256(patched),
                  byte_range_rva=[hex(fix.RVA), hex(fix.RVA+25)], changed_byte_count=len(changed),
                  original_null_parent_reproduced=baseline_fault, emulated_cases=len(cases),
                  native_executed_cases=native_count, supported_version_only=True,
                  idempotent=True, windows_loader_passed=True, instructions=instructions)
    if args.report_file:
        args.report_file.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
