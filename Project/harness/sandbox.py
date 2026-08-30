#!/usr/bin/env python3
"""Fail-closed Bubblewrap launcher for untrusted candidate evaluation.

Only explicitly selected files enter the mount namespace.  The repository,
authority journal, audit state, Git metadata, owner material, network, host PID
namespace, and the user's home directory are absent.  A fresh writable /tmp and
one controller-created output directory are the worker's only writable mounts.
"""

from __future__ import annotations

import glob
import ctypes
import errno
import functools
import os
import resource
import shutil
import signal
import site
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


class SandboxError(RuntimeError):
    pass


SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO_BASE = 0x00050000
NETWORK_SYSCALLS = (
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "sendto",
    "sendmsg",
    "sendmmsg",
    "recvfrom",
    "recvmsg",
    "recvmmsg",
    "shutdown",
)
CAPTURE_LIMIT_BYTES = 8 * 1024 * 1024
WORKER_FILE_LIMIT_BYTES = 128 * 1024 * 1024
# Unchanged from the original (256).  Measured on this host: a worker that
# imports torch, initialises CUDA and compiles + launches a Triton kernel
# peaks at 39 open descriptors, so this stays a real bound with headroom.
WORKER_OPEN_FILE_LIMIT = 256

# Read-only system runtime exposed in the jail.  Real directories are listed
# first so the merged-/usr symlinks that follow resolve to live content.  /bin
# and /sbin are mandatory, not cosmetic: Triton's CUDA driver bootstrap shells
# out to `ldconfig` to locate libcuda.so, so without them every Triton compile
# dies with "FileNotFoundError: [Errno 2] ... '/sbin/ldconfig'".
SYSTEM_ROOTS: tuple[str, ...] = (
    "/usr",
    "/etc",
    "/sys",
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
)
# The sbin directories are on PATH for the same reason.  Both spellings are
# listed so the lookup works whether the host merged /usr or not.
SANDBOX_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# RLIMIT_NPROC is charged to the real UID across the entire machine and counts
# every task (threads included), not just this process tree.  A fixed cap is
# therefore a cap on the host's total task count, not on the candidate.  The
# previous (128, 128) value made the clone() that bwrap performs immediately
# after preexec_fn fail on any normal desktop, so the sandbox could not start
# at all:  "bwrap: Creating new namespace failed: Resource temporarily
# unavailable".  (This UID owns ~1.3k tasks at rest.)  The cap is now derived
# from live usage -- double the current tasks plus this headroom -- and it is
# skipped entirely whenever that value is not comfortably reachable.  A
# sandbox that starts outranks a fork cap, and a runaway candidate is still
# bounded by --unshare-pid, --die-with-parent, the timeout with process-group
# kill, RLIMIT_NOFILE, RLIMIT_FSIZE and RLIMIT_CORE.
NPROC_HEADROOM = 4096


@dataclass(frozen=True)
class SandboxFiles:
    worker: Path
    candidate: Path
    official: Path
    shapes: Path
    request: Path
    output_dir: Path


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    command_shape: tuple[str, ...]


@dataclass(frozen=True)
class IsolatedMount:
    source: Path
    destination: str
    writable: bool = False


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise SandboxError(f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SandboxError(f"{label} must resolve to a regular non-symlink file")
    return resolved


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise SandboxError(f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise SandboxError(f"{label} must resolve to a non-symlink directory")
    return resolved


def _python_package_roots() -> list[Path]:
    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except AttributeError:
        pass
    roots: list[Path] = []
    for raw in candidates:
        path = Path(raw)
        if path.is_dir() and path.resolve() not in roots:
            roots.append(path.resolve())
    return roots


def _bind_if_exists(command: list[str], source: Path, destination: str) -> None:
    """Expose one host path in the jail, preserving its symlink-vs-directory shape.

    On merged-/usr distributions (Fedora here, also Arch and modern Debian) the
    root-level system paths are symlinks: /bin -> usr/bin, /sbin -> usr/sbin,
    /lib -> usr/lib, /lib64 -> usr/lib64.  ``Path.exists()`` follows the link,
    so a plain ``--ro-bind`` mounts the *target* as a second, real directory
    under the link's name -- a layout the host does not have, which breaks any
    tool that resolves or compares paths.  ``--symlink`` reproduces the host
    layout exactly, and a host with real directories still gets a bind, so this
    cannot regress either way.
    """
    if source.is_symlink():
        command.extend(["--symlink", os.readlink(source), destination])
        return
    if source.exists():
        command.extend(["--ro-bind", str(source), destination])


def _bind_system_roots(command: list[str]) -> None:
    """Bind the read-only system runtime.  No home directory is mounted.

    Both command builders call this so the two jails cannot drift apart.
    """
    for root in SYSTEM_ROOTS:
        _bind_if_exists(command, Path(root), root)


def _gpu_binds(command: list[str]) -> None:
    gpu_nodes = sorted(set(glob.glob("/dev/nvidia*") + glob.glob("/dev/dri/renderD*")))
    for index, node in enumerate(gpu_nodes):
        source = Path(node)
        if not source.exists():
            continue
        command.extend(["--dev-bind", str(source), f"/dev/gpu-node-{index}"])
        if source.name.startswith("nvidia"):
            command.extend(["--dev-bind", str(source), f"/dev/{source.name}"])


def build_command(
    files: SandboxFiles,
    worker_args: Iterable[str] = (),
    *,
    seccomp_fd: int | None = None,
) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxError("bubblewrap is required; unsandboxed fallback is forbidden")

    worker = _regular_file(files.worker, "worker")
    candidate = _regular_file(files.candidate, "candidate")
    official = _regular_file(files.official, "official benchmark")
    shapes = _regular_file(files.shapes, "shapes")
    request = _regular_file(files.request, "request")
    output = _directory(files.output_dir, "output directory")

    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--dir",
        "/output",
        "--dir",
        "/opt",
    ]
    if seccomp_fd is not None:
        command.extend(["--seccomp", str(seccomp_fd)])
    else:
        # No syscall ban means the network namespace must do the isolating.
        # If the kernel refuses it bwrap dies here, which is fail-closed.
        command.append("--unshare-net")

    # System runtime and loader.  No home directory is mounted.
    _bind_system_roots(command)

    python_paths: list[str] = []
    for index, package_root in enumerate(_python_package_roots()):
        destination = f"/opt/python-site-{index}"
        command.extend(["--ro-bind", str(package_root), destination])
        python_paths.append(destination)

    # Minimal GPU devices.  They are absent on CPU-only hosts, which is useful
    # for static/adversarial tests and fails naturally for a GPU request.
    _gpu_binds(command)

    command.extend(
        [
            "--ro-bind",
            str(worker),
            "/work/worker.py",
            "--ro-bind",
            str(candidate),
            "/work/candidate.py",
            "--ro-bind",
            str(official),
            "/work/official.py",
            "--ro-bind",
            str(shapes),
            "/work/shapes.json",
            "--ro-bind",
            str(request),
            "/work/request.json",
            "--bind",
            str(output),
            "/output",
            "--setenv",
            "HOME",
            "/nonexistent",
            "--setenv",
            "PATH",
            SANDBOX_PATH,
            "--setenv",
            "PYTHONPATH",
            ":".join(python_paths),
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "TRITON_CACHE_DIR",
            "/tmp/triton-cache",
            "--setenv",
            "CUDA_CACHE_PATH",
            "/tmp/cuda-cache",
            "--chdir",
            "/work",
            "/usr/bin/python3",
            "/work/worker.py",
            "--request",
            "/work/request.json",
            "--candidate",
            "/work/candidate.py",
            "--official",
            "/work/official.py",
            "--shapes",
            "/work/shapes.json",
            "--output",
            "/output",
            *worker_args,
        ]
    )
    return command


def _uid_task_count() -> int | None:
    """Count the tasks currently charged to the real UID, or None if unknown."""
    uid = os.getuid()
    total = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for name in entries:
        if not name.isdigit():
            continue
        try:
            if os.stat(f"/proc/{name}").st_uid != uid:
                continue
            total += len(os.listdir(f"/proc/{name}/task"))
        except OSError:
            continue  # the process exited mid-scan; it is not ours to count
    return total or None


def _worker_nproc_limit() -> int | None:
    """Defensive RLIMIT_NPROC value for the worker, or None for no cap.

    Computed in the parent, before fork, so the child's preexec_fn only has to
    call setrlimit.  See NPROC_HEADROOM for why a fixed cap is wrong.
    """
    used = _uid_task_count()
    if used is None:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    wanted = used * 2 + NPROC_HEADROOM
    if hard != resource.RLIM_INFINITY and wanted > hard:
        return None  # cannot raise past the hard limit; leave the cap alone
    if soft != resource.RLIM_INFINITY and wanted >= soft:
        return None  # the inherited limit is already tighter than anything safe
    return wanted


def _limit_worker(nproc_limit: int | None = None) -> None:
    """preexec_fn for the sandbox child: per-process bounds only.

    Everything set here is charged to this process alone, except NPROC, which
    is machine-wide per UID and is therefore passed in already computed (None
    means: set no cap, because any value would risk breaking process creation).
    """
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        (WORKER_OPEN_FILE_LIMIT, WORKER_OPEN_FILE_LIMIT),
    )
    if nproc_limit is not None:
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc_limit, nproc_limit))
    # Evidence output is bounded.  GPU allocations are not governed by AS and
    # are controlled separately by request shape/memory checks.
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (WORKER_FILE_LIMIT_BYTES, WORKER_FILE_LIMIT_BYTES),
    )


_NETNS_SUPPORTED: bool | None = None


def _probe_netns() -> bool:
    """Fork once and find out whether this user may create a network namespace."""
    if not hasattr(os, "unshare") or not hasattr(os, "CLONE_NEWNET"):
        return False
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid == 0:  # child: exit 0 only if both namespaces were created
        code = 1
        try:
            os.unshare(os.CLONE_NEWUSER)  # grants CAP_SYS_ADMIN inside the new user ns
            os.unshare(os.CLONE_NEWNET)
            code = 0
        except OSError:
            code = 1
        finally:
            os._exit(code)
    try:
        _, status = os.waitpid(pid, 0)
    except OSError:
        return False
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def _netns_available() -> bool:
    """Cached result of :func:`_probe_netns`."""
    global _NETNS_SUPPORTED
    if _NETNS_SUPPORTED is None:
        _NETNS_SUPPORTED = _probe_netns()
    return _NETNS_SUPPORTED


def _network_seccomp_fd() -> int | None:
    """Network isolation for the worker: fd to pass to --seccomp, or None.

    ``--unshare-net`` is the primary mechanism: the worker gets a private
    network namespace with nothing but a down loopback, so there is no route
    off the box and abstract unix sockets (X11, dbus) are unreachable too.
    It is preferred over the syscall ban because the NVIDIA user-mode driver
    needs local socket syscalls: with the ban in force cuInit() fails with 304
    (CUDA_ERROR_OPERATING_SYSTEM) and no GPU work can run at all -- proven by
    bisection on this host (no seccomp: cuInit 0; seccomp: cuInit 304).
    The ban stays as the fallback for managed hosts that forbid creating a
    network namespace even inside a user namespace; such a host cannot run GPU
    candidates under this sandbox, and that is the fail-closed direction.
    """
    if _netns_available():
        return None
    return _network_deny_filter()


def _network_deny_filter() -> int:
    """Return a sealed memfd containing a libseccomp BPF network deny filter.

    Fallback only -- see :func:`_network_seccomp_fd`.  Blocking socket creation
    and all connection/transfer syscalls is enforceable by the kernel, but it
    also blocks the local socket use of the CUDA driver.
    """
    try:
        library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise SandboxError("libseccomp is required for network isolation") from exc
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    # Variadic function: these are the fixed arguments; arg_cnt is always zero.
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.seccomp_export_bpf.restype = ctypes.c_int
    context = library.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise SandboxError("seccomp_init failed")
    fd = -1
    try:
        deny = SCMP_ACT_ERRNO_BASE | errno.EPERM
        for name in NETWORK_SYSCALLS:
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                continue
            rc = library.seccomp_rule_add(context, deny, number, 0)
            if rc != 0:
                raise SandboxError(f"cannot add seccomp rule for {name}: {rc}")
        fd = os.memfd_create("candidate-network-deny", os.MFD_CLOEXEC)
        if library.seccomp_export_bpf(context, fd) != 0:
            raise SandboxError("cannot export seccomp network filter")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except Exception:
        if fd >= 0:
            os.close(fd)
        raise
    finally:
        library.seccomp_release(context)


def run_sandbox(
    files: SandboxFiles,
    *,
    worker_args: Iterable[str] = (),
    timeout_seconds: int = 1800,
) -> SandboxResult:
    if timeout_seconds < 1 or timeout_seconds > 24 * 3600:
        raise SandboxError("sandbox timeout must be between 1 second and 24 hours")
    seccomp_fd = _network_seccomp_fd()
    command = build_command(files, worker_args, seccomp_fd=seccomp_fd)
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                preexec_fn=functools.partial(_limit_worker, _worker_nproc_limit()),
                pass_fds=() if seccomp_fd is None else (seccomp_fd,),
            )
        finally:
            if seccomp_fd is not None:
                os.close(seccomp_fd)
        timed_out = False
        try:
            process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        stdout_size = stdout_file.seek(0, os.SEEK_END)
        stderr_size = stderr_file.seek(0, os.SEEK_END)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(CAPTURE_LIMIT_BYTES)
        stderr = stderr_file.read(CAPTURE_LIMIT_BYTES)
        if stdout_size > CAPTURE_LIMIT_BYTES or stderr_size > CAPTURE_LIMIT_BYTES:
            returncode = 125
            stderr += b"\nSANDBOX_REFUSED: worker output exceeded capture limit\n"
        else:
            returncode = process.returncode
    return SandboxResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        command_shape=tuple(
            "<host-path>" if part.startswith("/") and part not in {
                "/usr/bin/python3", "/work/worker.py", "/work/request.json",
                "/work/candidate.py", "/work/official.py", "/work/shapes.json",
                "/output", "/work", "/tmp", "/proc", "/dev", "/opt",
            } else part
            for part in command
        ),
    )


def _safe_destination(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or ".." in pure.parts
        or str(pure) != value
        or value in {"/", "/usr", "/lib", "/lib64", "/etc", "/sys", "/dev", "/proc"}
    ):
        raise SandboxError(f"unsafe isolated mount destination: {value!r}")
    return value


def run_isolated_command(
    *,
    mounts: Iterable[IsolatedMount],
    argv: Iterable[str],
    cwd: str,
    timeout_seconds: int = 1800,
) -> SandboxResult:
    """Run a trusted evaluator around untrusted candidate bytes.

    This is used for side-shape and profiler workers whose virtual directory
    layout is more complex than :class:`SandboxFiles`.  The same namespace,
    network, capability, environment, resource, timeout, and bounded-capture
    rules apply.  Callers supply only explicit mounts; the repository itself
    is never mounted.
    """
    if timeout_seconds < 1 or timeout_seconds > 24 * 3600:
        raise SandboxError("sandbox timeout must be between 1 second and 24 hours")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxError("bubblewrap is required; unsandboxed fallback is forbidden")
    mount_list = list(mounts)
    if not mount_list:
        raise SandboxError("isolated command needs explicit mounts")
    destinations: set[str] = set()
    resolved_mounts: list[tuple[Path, str, bool]] = []
    parent_dirs = {"/work", "/output", "/opt", "/sandbox"}
    for mount in mount_list:
        destination = _safe_destination(mount.destination)
        if destination in destinations:
            raise SandboxError(f"duplicate isolated destination: {destination}")
        destinations.add(destination)
        if mount.source.is_symlink():
            raise SandboxError("isolated mount sources cannot be symlinks")
        source = mount.source.resolve(strict=True)
        if not source.is_file() and not source.is_dir():
            raise SandboxError("isolated mount source must be a file or directory")
        if mount.writable and not source.is_dir():
            raise SandboxError("only isolated directories may be writable")
        resolved_mounts.append((source, destination, mount.writable))
        parent = PurePosixPath(destination).parent
        while str(parent) not in {"/", "."}:
            parent_dirs.add(str(parent))
            parent = parent.parent
    cwd = _safe_destination(cwd)
    command = [
        bwrap,
        "--die-with-parent", "--new-session", "--unshare-user",
        "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--cap-drop", "ALL", "--clearenv", "--proc", "/proc",
        "--dev", "/dev", "--tmpfs", "/tmp",
    ]
    seccomp_fd = _network_seccomp_fd()
    if seccomp_fd is not None:
        command.extend(["--seccomp", str(seccomp_fd)])
    else:
        command.append("--unshare-net")
    for directory in sorted(parent_dirs, key=lambda value: (value.count("/"), value)):
        command.extend(["--dir", directory])
    _bind_system_roots(command)
    python_paths: list[str] = []
    for index, package_root in enumerate(_python_package_roots()):
        destination = f"/opt/python-site-{index}"
        command.extend(["--ro-bind", str(package_root), destination])
        python_paths.append(destination)
    _gpu_binds(command)
    for source, destination, writable in resolved_mounts:
        command.extend(["--bind" if writable else "--ro-bind", str(source), destination])
    arguments = list(argv)
    if not arguments or any(not isinstance(item, str) or "\x00" in item for item in arguments):
        if seccomp_fd is not None:
            os.close(seccomp_fd)
        raise SandboxError("isolated argv is malformed")
    command.extend([
        "--setenv", "HOME", "/nonexistent",
        "--setenv", "PATH", SANDBOX_PATH,
        "--setenv", "PYTHONPATH", ":".join(python_paths),
        "--setenv", "PYTHONNOUSERSITE", "1",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "TRITON_CACHE_DIR", "/tmp/triton-cache",
        "--setenv", "CUDA_CACHE_PATH", "/tmp/cuda-cache",
        "--chdir", cwd,
        *arguments,
    ])
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                preexec_fn=functools.partial(_limit_worker, _worker_nproc_limit()),
                pass_fds=() if seccomp_fd is None else (seccomp_fd,),
            )
        finally:
            if seccomp_fd is not None:
                os.close(seccomp_fd)
        timed_out = False
        try:
            process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        stdout_size = stdout_file.seek(0, os.SEEK_END)
        stderr_size = stderr_file.seek(0, os.SEEK_END)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(CAPTURE_LIMIT_BYTES)
        stderr = stderr_file.read(CAPTURE_LIMIT_BYTES)
        returncode = process.returncode
        if stdout_size > CAPTURE_LIMIT_BYTES or stderr_size > CAPTURE_LIMIT_BYTES:
            returncode = 125
            stderr += b"\nSANDBOX_REFUSED: worker output exceeded capture limit\n"
    return SandboxResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        command_shape=tuple("<host-path>" if item.startswith("/") else item for item in command),
    )
