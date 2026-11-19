"""Small cross-platform exclusive file locks for application coordination."""

import errno
import os
import time


if os.name == "nt":
    import msvcrt
else:
    import fcntl


def acquire_file_lock(file_handle, *, blocking=True):
    """Lock an already-open file handle until it is unlocked or closed.

    Raises BlockingIOError when a non-blocking lock is already held.
    """
    if os.name == "nt":
        _acquire_windows_lock(file_handle, blocking=blocking)
        return

    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    fcntl.flock(file_handle.fileno(), operation)


def release_file_lock(file_handle):
    """Release a lock acquired with :func:`acquire_file_lock`."""
    if os.name == "nt":
        file_handle.seek(0)
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def _acquire_windows_lock(file_handle, *, blocking):
    # msvcrt locks a byte range starting at the current file position. Ensure
    # byte zero exists and use it as the shared lock range for all processes.
    file_handle.seek(0, os.SEEK_END)
    if file_handle.tell() == 0:
        file_handle.write(b"\0")
        file_handle.flush()

    while True:
        file_handle.seek(0)
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if not _is_windows_lock_conflict(exc):
                raise
            if not blocking:
                raise BlockingIOError(
                    errno.EWOULDBLOCK,
                    "The file is already locked",
                    getattr(file_handle, "name", None)
                ) from exc
            time.sleep(0.1)


def _is_windows_lock_conflict(error):
    conflict_errnos = {
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EDEADLK", -1)
    }
    return (
        error.errno in conflict_errnos
        or getattr(error, "winerror", None) in {33, 36}
    )
