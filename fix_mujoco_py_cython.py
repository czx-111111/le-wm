#!/usr/bin/env python3
"""修复 mujoco-py 与 Cython 的兼容问题，并验证 import。

用法（在 le-wm venv 里）:
  python fix_mujoco_py_cython.py

最稳妥（对齐 DINO-WM）:
  pip install 'cython==0.29.37'
  python fix_mujoco_py_cython.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _find_mujoco_root() -> Path | None:
    candidates = [
        ROOT / "mujoco210" / "mujoco210",
        ROOT / "mujoco210",
        Path.home() / ".mujoco" / "mujoco210",
    ]
    for c in candidates:
        if (c / "bin").is_dir():
            return c.resolve()
    return None


def _setup_mujoco_path():
    """设置 MuJoCo 路径；若需更新 LD_LIBRARY_PATH 则 re-exec（运行时改环境对 dlopen 无效）。"""
    mujoco_root = _find_mujoco_root()
    if mujoco_root is None:
        print("[WARN] 未找到 mujoco210，import 时可能仍失败")
        return

    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(mujoco_root)
    bin_dir = str(mujoco_root / "bin")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(os.pathsep) if p]
    extras = [bin_dir, "/usr/lib/nvidia", "/usr/lib/x86_64-linux-gnu"]
    new_parts = []
    for p in extras + parts:
        if p and p not in new_parts:
            new_parts.append(p)
    new_ld = os.pathsep.join(new_parts)
    need_reexec = bin_dir not in parts

    os.environ["LD_LIBRARY_PATH"] = new_ld
    print(f"[INFO] MUJOCO_PY_MUJOCO_PATH={mujoco_root}")
    print(f"[INFO] LD_LIBRARY_PATH 含 mujoco bin: {bin_dir}")

    if need_reexec and os.environ.get("_LEWM_MUJOCO_REEXEC") != "1":
        os.environ["_LEWM_MUJOCO_REEXEC"] = "1"
        print("[INFO] re-exec：使新的 LD_LIBRARY_PATH 对动态链接器生效...")
        os.execv(sys.executable, [sys.executable, *sys.argv])


def _preload_mujoco_libs():
    """双保险：绝对路径预加载，避免仍找不到 libglewegl.so。"""
    import ctypes

    mujoco_root = _find_mujoco_root()
    if mujoco_root is None:
        return
    bin_dir = mujoco_root / "bin"
    for name in (
        "libglewegl.so",
        "libglew.so",
        "libglfw.so.3",
        "libmujoco210.so",
    ):
        lib = bin_dir / name
        if not lib.is_file():
            continue
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            print(f"[INFO] preload {lib.name}")
        except OSError as e:
            print(f"[WARN] preload {lib.name} 失败: {e}")


def _fix_package(pkg: Path, cython_major: int, cython_ver: str) -> None:
    builder = pkg / "builder.py"
    cymj = pkg / "cymj.pyx"

    # 回退 legacy_implicit_noexcept（部分 Cython 不支持该 directive）
    if builder.is_file():
        text = builder.read_text(encoding="utf-8")
        if "legacy_implicit_noexcept" in text:
            text2, n = re.subn(
                r"dist\.ext_modules\s*=\s*cythonize\(\s*\[self\.extension\]\s*,"
                r"\s*compiler_directives\s*=\s*\{['\"]legacy_implicit_noexcept['\"]\s*:\s*True\}\s*\)",
                "dist.ext_modules = cythonize([self.extension])",
                text,
                count=1,
            )
            if n:
                builder.write_text(text2, encoding="utf-8")
                print("[INFO] 已回退 builder.py 的 legacy_implicit_noexcept 补丁")
            else:
                raise SystemExit(
                    "builder.py 含 legacy_implicit_noexcept 但未能自动回退，"
                    "请手动改回: dist.ext_modules = cythonize([self.extension])"
                )

    if cython_major < 3:
        print(f"[INFO] Cython {cython_ver} < 3，无需改 cymj.pyx")
        return

    if not cymj.is_file():
        raise SystemExit(f"找不到 {cymj}")

    pyx = cymj.read_text(encoding="utf-8")
    new_pyx = pyx
    new_pyx = re.sub(
        r"cdef void c_warning_callback\(const char \*msg\)(?! noexcept) with gil:",
        "cdef void c_warning_callback(const char *msg) noexcept with gil:",
        new_pyx,
        count=1,
    )
    new_pyx = re.sub(
        r"cdef void c_error_callback\(const char \*msg\)(?! noexcept) with gil:",
        "cdef void c_error_callback(const char *msg) noexcept with gil:",
        new_pyx,
        count=1,
    )
    if new_pyx != pyx:
        cymj.write_text(new_pyx, encoding="utf-8")
        for pattern in ("cymj.c", "cymj.cpp", "cymj*.so"):
            for p in pkg.glob(pattern):
                try:
                    p.unlink()
                    print(f"[INFO] 删除旧编译产物: {p.name}")
                except OSError:
                    pass
        print(f"[INFO] 已修补 cymj.pyx（noexcept），Cython={cython_ver}")
    elif "noexcept with gil" in pyx:
        print("[INFO] cymj.pyx 已含 noexcept")
    else:
        raise SystemExit(
            "未能修补 cymj.pyx。请改用: pip install 'cython==0.29.37'"
        )


def _check_glew_headers() -> None:
    candidates = [
        Path("/usr/include/GL/glew.h"),
        Path("/usr/local/include/GL/glew.h"),
        Path("/data/czx/le-wm/deps/sysroot/usr/include/GL/glew.h"),
        Path("/data/czx/le-wm/deps/glew/include/GL/glew.h"),
        ROOT.parent / "deps" / "sysroot" / "usr" / "include" / "GL" / "glew.h",
        ROOT.parent / "deps" / "glew" / "include" / "GL" / "glew.h",
    ]
    found = next((p for p in candidates if p.is_file()), None)
    if found is not None:
        include_root = found.parent.parent  # .../include
        lib_root = include_root.parent / "lib" / "x86_64-linux-gnu"
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            cur = os.environ.get(key, "")
            parts = [p for p in cur.split(os.pathsep) if p]
            if str(include_root) not in parts:
                os.environ[key] = os.pathsep.join([str(include_root), *parts])
        if lib_root.is_dir():
            for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
                cur = os.environ.get(key, "")
                parts = [p for p in cur.split(os.pathsep) if p]
                if str(lib_root) not in parts:
                    os.environ[key] = os.pathsep.join([str(lib_root), *parts])
        os.environ["CFLAGS"] = f"-I{include_root} " + os.environ.get("CFLAGS", "")
        print(f"[INFO] 找到 GLEW 头文件: {found}")
        return
    raise SystemExit(
        "缺少 GLEW 开发头文件 (GL/glew.h)，mujoco_py 无法编译 eglshim.c。\n"
        "请在机器上安装系统依赖后重试:\n"
        "  sudo apt-get update\n"
        "  sudo apt-get install -y libglew-dev libglfw3-dev libosmesa6-dev\n"
        "若无 sudo，请联系管理员安装上述包。"
    )


def main():
    _setup_mujoco_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    _check_glew_headers()

    try:
        import Cython

        cython_ver = str(Cython.__version__)
        cython_major = int(cython_ver.split(".")[0])
    except Exception as e:
        raise SystemExit(f"无法 import Cython: {e}") from e
    print(f"[INFO] Cython={cython_ver}")

    spec = importlib.util.find_spec("mujoco_py")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("未安装 mujoco_py")
    pkg = Path(list(spec.submodule_search_locations)[0])
    print(f"[INFO] mujoco_py package: {pkg}")

    _fix_package(pkg, cython_major, cython_ver)

    print("[INFO] 验证 import mujoco_py（首次可能重新编译，稍等）...")
    _preload_mujoco_libs()
    # 避免缓存到旧失败模块
    for name in list(sys.modules):
        if name.startswith("mujoco_py"):
            del sys.modules[name]
    import mujoco_py  # noqa: F401

    print(f"[INFO] OK: {mujoco_py.__file__}")


if __name__ == "__main__":
    main()
