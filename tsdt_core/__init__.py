"""tsdt_core: headless engine for the Time-Series Data Trimmer.

UI-agnostic data model, adaptive ingestion, and session persistence.
Everything here runs without Qt, so it can be reused from CLI tools,
notebooks, and batch pipelines.
"""
__version__ = "1.0.0"

from tsdt_core.core import CoreDataModel  # noqa: E402
from tsdt_core.ingest import IngestReport, smart_read, sniff_csv  # noqa: E402
from tsdt_core.models import AnnotationSegment, OperationRecord  # noqa: E402
from tsdt_core.session_io import Session, load_session, save_session  # noqa: E402

__all__ = [
    "AnnotationSegment",
    "CoreDataModel",
    "IngestReport",
    "OperationRecord",
    "Session",
    "load_session",
    "save_session",
    "smart_read",
    "sniff_csv",
    "__version__",
]
