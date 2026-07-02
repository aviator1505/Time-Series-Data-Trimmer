"""tsdt_core: headless engine for the Time-Series Data Trimmer.

UI-agnostic data model, adaptive ingestion, and session persistence.
Everything here runs without Qt, so it can be reused from CLI tools,
notebooks, and batch pipelines.
"""
from tsdt_core.core import CoreDataModel
from tsdt_core.ingest import IngestReport, smart_read, sniff_csv
from tsdt_core.models import AnnotationSegment, OperationRecord

__all__ = [
    "AnnotationSegment",
    "CoreDataModel",
    "IngestReport",
    "OperationRecord",
    "smart_read",
    "sniff_csv",
]
