"""Exclusive OS file lock (standard library only): the job runner's queue lock and the registry's write lock."""
import os
import time


def acquire_lock(path, wait=0.0):
    """Exclusive OS lock on `path` (msvcrt.locking on Windows, flock elsewhere), or None if another process
    holds it after `wait` seconds. The lock dies with the process; the file staying behind is normal."""
    f = open(path, "a+b")
    deadline = time.time() + wait
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.time() >= deadline:
                f.close()
                return None
            time.sleep(0.5)
