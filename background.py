"""Run heavy computations off the UI thread via QThreadPool.

Workers must be pure compute (no model mutation, no widget access).
Results are delivered back on the UI thread through queued signal
connections, so `on_finished`/`on_error` handlers may safely touch
models and widgets.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6 import QtCore

# Keep strong references to in-flight jobs: QThreadPool owns the C++ side,
# but the Python wrapper (and its closures) must not be garbage-collected
# while the job runs.
_active_jobs: set[BackgroundJob] = set()


class _JobSignals(QtCore.QObject):
    finished = QtCore.Signal(object)  # result of fn
    error = QtCore.Signal(object)  # the raised exception


class BackgroundJob(QtCore.QRunnable):
    """QRunnable wrapping a callable; emits finished(result) or error(exc)."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = _JobSignals()
        self.cancelled = False  # cooperative: result is dropped, not interrupted

    def cancel(self) -> None:
        """Mark the job cancelled: it still runs, but emits nothing."""
        self.cancelled = True

    @QtCore.Slot()
    def run(self) -> None:  # executed on a pool thread
        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:  # deliver any failure to the UI thread
            if not self.cancelled:
                self.signals.error.emit(exc)
        else:
            if not self.cancelled:
                self.signals.finished.emit(result)
        finally:
            _active_jobs.discard(self)


def run_in_background(
    fn: Callable[..., Any],
    *args: Any,
    on_finished: Callable[[Any], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    **kwargs: Any,
) -> BackgroundJob:
    """Start fn(*args, **kwargs) on the global thread pool.

    on_finished receives fn's return value; on_error receives the exception.
    Both run on the UI thread. Returns the job (call .cancel() to drop the
    result of a job that is no longer wanted).
    """
    job = BackgroundJob(fn, *args, **kwargs)
    if on_finished is not None:
        job.signals.finished.connect(on_finished)
    if on_error is not None:
        job.signals.error.connect(on_error)
    _active_jobs.add(job)
    QtCore.QThreadPool.globalInstance().start(job)
    return job
