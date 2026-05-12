from langgraph.graph import END, StateGraph

from config.settings import GRAPH_RECURSION_LIMIT
from processing_pipeline.nodes.aggregation_agent import (
    aggregation_agent,
    chapter_aggregator,
    chapter_aggregator_dispatch,
    dispatch_chapters_to_aggregators,
    dispatch_sections_to_aggregators,
    section_aggregator,
)
from processing_pipeline.nodes.extraction import extraction
from processing_pipeline.nodes.preprocessing import preprocessing
from processing_pipeline.nodes.segment_analyzer_agent import (
    dispatch_segments_to_workers,
    segment_analyzer_agent,
    segment_analyzer_worker,
)
from processing_pipeline.nodes.segmentation import segmentation
from processing_pipeline.state import DocumentProcessingState


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    workflow = StateGraph(DocumentProcessingState)

    # Deterministic ingestion pipeline
    workflow.add_node("extraction", extraction)
    workflow.add_node("preprocessing", preprocessing)
    workflow.add_node("segmentation", segmentation)

    # Local-understanding fan-out (per segment)
    workflow.add_node("segment_analyzer_agent", segment_analyzer_agent)
    workflow.add_node("segment_analyzer_worker", segment_analyzer_worker)

    # Section reduction fan-out
    workflow.add_node("aggregation_agent", aggregation_agent)
    workflow.add_node("section_aggregator", section_aggregator)

    # Chapter reduction fan-out
    workflow.add_node("chapter_aggregator_dispatch", chapter_aggregator_dispatch)
    workflow.add_node("chapter_aggregator", chapter_aggregator)

    workflow.set_entry_point("extraction")
    workflow.add_edge("extraction", "preprocessing")
    workflow.add_edge("preprocessing", "segmentation")
    workflow.add_edge("segmentation", "segment_analyzer_agent")

    # Fan out segments → segment_analyzer_worker (joins at aggregation_agent)
    workflow.add_conditional_edges(
        "segment_analyzer_agent",
        dispatch_segments_to_workers,
        ["segment_analyzer_worker"],
    )
    workflow.add_edge("segment_analyzer_worker", "aggregation_agent")

    # Fan out sections → section_aggregator (joins at chapter_aggregator_dispatch)
    workflow.add_conditional_edges(
        "aggregation_agent",
        dispatch_sections_to_aggregators,
        ["section_aggregator"],
    )
    workflow.add_edge("section_aggregator", "chapter_aggregator_dispatch")

    # Fan out chapters → chapter_aggregator (joins at END)
    workflow.add_conditional_edges(
        "chapter_aggregator_dispatch",
        dispatch_chapters_to_aggregators,
        ["chapter_aggregator"],
    )
    workflow.add_edge("chapter_aggregator", END)

    return workflow.compile()


app = build_graph().with_config({"recursion_limit": GRAPH_RECURSION_LIMIT})
