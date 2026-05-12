import operator
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DocumentProcessingState(TypedDict, total=False):
    # Identity
    run_id: str
    document_id: str
    document_name: str
    user_id: Optional[str]
    doc_id: Optional[str]

    # Deterministic node outputs
    extracted_data: Optional[dict[str, Any]]
    preprocessed_data: Optional[dict[str, Any]]
    toc: Optional[list[dict[str, Any]]]
    segments: Optional[list[dict[str, Any]]]

    # Per-Send transient payloads (one of these is populated per worker branch)
    segment: Optional[dict[str, Any]]            # segment_analyzer_worker
    section_name: Optional[str]                  # section_aggregator
    section_segments: Optional[list[dict]]       # section_aggregator
    chapter_name: Optional[str]                  # chapter_aggregator
    chapter_sections: Optional[list[dict]]       # chapter_aggregator

    # Reduction results, accumulated across parallel Send branches
    segment_analyses: Annotated[list[dict[str, Any]], operator.add]
    section_analyses: Annotated[list[dict[str, Any]], operator.add]
    chapter_analyses: Annotated[list[dict[str, Any]], operator.add]
