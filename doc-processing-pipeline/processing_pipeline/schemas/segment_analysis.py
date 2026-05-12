import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _decode_list_field(value: Any) -> Any:
    """Field-level before-validator: OpenAI structured output occasionally
    emits a `list[...]` field as a JSON-encoded string instead of a real
    array. Detect that shape and decode it before pydantic type-checks.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
            except ValueError:
                return value
            if isinstance(decoded, list):
                return decoded
    return value

Confidence = Literal["high", "medium", "low"]
Sentiment = Literal["positive", "negative", "neutral", "mixed"]
Polarity = Literal["positive", "negative", "neutral"]


class Entity(BaseModel):
    name: str = Field(
        description="Canonical name of the entity as it appears in the segment (e.g., 'ICICI Bank', 'P. K. Sinha')."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other surface forms used to refer to this entity in the same segment (e.g., ['The Bank', 'we']). Empty list if none.",
    )
    type: str = Field(
        description="Entity category: one of 'organization', 'person', 'product', 'location', 'regulation', 'metric', 'event', or another short noun phrase that best fits.",
    )
    role: str = Field(
        description="Role or function the entity plays in this segment (e.g., 'Outgoing Chairman of the Board', 'Issuer of the report').",
    )
    context: str = Field(
        description="Why this entity matters in this segment — one short sentence grounding its relevance to the surrounding text.",
    )


class Metric(BaseModel):
    name: str = Field(
        description="Short label for the metric (e.g., 'CSR Beneficiaries', 'Profit After Tax', 'Net Interest Margin')."
    )
    value_text: str = Field(
        description="The metric value exactly as written in the document, including currency / unit markers (e.g., '₹408.88 billion', 'over 12 million', '4.53%').",
    )
    numeric_value: Optional[float] = Field(
        default=None,
        description="Best-effort numeric value in base units (e.g., 12 million → 12000000, ₹408.88 billion → 408880000000). Null if the text is non-numeric or ambiguous.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement (e.g., 'people', 'INR', 'percent', 'count'). Null if not applicable.",
    )
    time_period: Optional[str] = Field(
        default=None,
        description="Time period the metric covers (e.g., 'FY24', 'as at March 31, 2024', 'fiscal 2024'). Null if not stated.",
    )
    comparison: Optional[str] = Field(
        default=None,
        description="Comparison or trend context if stated (e.g., 'up from 9% previous year', '34% y-o-y'). Null if not stated.",
    )
    entity_ref: Optional[str] = Field(
        default=None,
        description="Name of the key_entity this metric is about. Must match a name in key_entities. Null if the metric is generic.",
    )
    supporting_quote: str = Field(
        description="A short verbatim quote from the segment that contains this metric.",
    )


class Claim(BaseModel):
    text: str = Field(
        description="One-sentence paraphrase of the claim being made by the document (e.g., 'The Bank's CSR initiatives have benefited over 12 million people')."
    )
    claim_type: str = Field(
        description="Short type label, e.g., 'historical_fact', 'forward_looking', 'governance_position', 'strategic_position', 'market_position', 'performance_claim', 'risk_assessment'.",
    )
    entity_refs: list[str] = Field(
        default_factory=list,
        description="Names of entities or metrics involved in this claim. Each value should match a key_entities.name or a metrics.name.",
    )
    polarity: Polarity = Field(
        description="Whether the claim casts the subject in a positive, negative, or neutral light.",
    )
    temporal_scope: str = Field(
        description="When the claim applies: 'FY24', 'historical', 'forward_looking', 'ongoing', or an explicit period.",
    )
    supporting_quote: str = Field(
        description="Verbatim quote from the segment that supports this claim.",
    )
    confidence: Confidence = Field(
        description="Your confidence that this claim is faithfully extracted from the text (not inferred or hallucinated).",
    )


class Decision(BaseModel):
    text: str = Field(
        description="One-sentence description of a decision, appointment, approval, or action announced in the segment."
    )
    decision_type: str = Field(
        description="Short type label, e.g., 'appointment', 'strategic', 'governance', 'financial', 'policy', 'operational', 'transaction'.",
    )
    actors: list[str] = Field(
        default_factory=list,
        description="Entities making the decision (e.g., ['Board of Directors']). Match key_entities.name when possible.",
    )
    affected_entities: list[str] = Field(
        default_factory=list,
        description="Entities affected by the decision. Match key_entities.name when possible.",
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Stated effective / announcement date (e.g., 'July 1, 2024', 'FY24'). Null if not stated.",
    )
    supporting_quote: str = Field(
        description="Verbatim quote from the segment that supports this decision.",
    )


class Risk(BaseModel):
    text: str = Field(
        description="One-sentence description of a risk, threat, or uncertainty mentioned in the segment."
    )
    risk_category: str = Field(
        description="Category, e.g., 'geopolitical', 'credit', 'market', 'operational', 'regulatory', 'cyber', 'climate', 'liquidity', 'reputational', 'strategic'.",
    )
    severity: Optional[Confidence] = Field(
        default=None,
        description="Severity inferred from the segment ('high' / 'medium' / 'low'). Null if not characterized.",
    )
    affected_entities: list[str] = Field(
        default_factory=list,
        description="Entities exposed to this risk. Match key_entities.name when possible.",
    )
    mitigation_mentioned: Optional[str] = Field(
        default=None,
        description="Mitigation, hedge, or response the document attributes to this risk. Null if not stated.",
    )
    supporting_quote: str = Field(
        description="Verbatim quote from the segment that supports this risk.",
    )


class Contradiction(BaseModel):
    description: str = Field(
        description="What is internally inconsistent within this segment — describe the conflict in one sentence."
    )
    statements: list[str] = Field(
        description="The two or more statements from the segment that are in tension, paraphrased.",
    )
    supporting_quotes: list[str] = Field(
        description="Verbatim quotes from the segment corresponding to each statement (same order as `statements`).",
    )


class SegmentAnalysis(BaseModel):
    """Structured local-understanding analysis of ONE segment of an annual report.

    Use ONLY the text of the provided segment as evidence. Do not introduce facts
    that are not present in the segment. Every metric, claim, decision, risk, and
    contradiction MUST include a verbatim `supporting_quote` taken directly from
    the segment text.
    """

    _decode_list_fields = field_validator(
        "topics",
        "key_entities",
        "metrics",
        "key_claims",
        "decisions",
        "risks",
        "internal_contradictions",
        "salient_quotes",
        "notable_other",
        mode="before",
    )(_decode_list_field)

    ocr_confidence: Confidence = Field(
        description="Your overall confidence that the OCR text is clean and readable in this segment. 'high' = clean; 'medium' = some garbled tokens or layout issues; 'low' = heavily corrupted or fragmented.",
    )
    has_table: bool = Field(
        description="True if the segment contains at least one markdown table or tabular data block (look for '|' separators or aligned numeric columns).",
    )
    has_figure: bool = Field(
        description="True if the segment references at least one figure, chart, or image (look for markdown image syntax `![...](...)` or phrases like 'bar chart showing...').",
    )
    summary: str = Field(
        description="A faithful 3–6 sentence summary of the segment that preserves the key facts, named entities, and figures. No new information, no opinions.",
    )
    topics: list[str] = Field(
        description="3–8 short topic tags (Title Case noun phrases) covering the main themes of the segment (e.g., 'Corporate Social Responsibility', 'Leadership Transition').",
    )
    sentiment: Sentiment = Field(
        description="Overall tone of the segment as written by the issuer: 'positive', 'negative', 'neutral', or 'mixed'.",
    )
    key_entities: list[Entity] = Field(
        description="Salient named entities referenced in the segment (organizations, people, products, regulations, locations). Skip generic terms.",
    )
    metrics: list[Metric] = Field(
        description="Quantitative metrics, KPIs, or counts stated in the segment. Include both monetary and non-monetary measures. Empty list if none.",
    )
    key_claims: list[Claim] = Field(
        description="Substantive claims the issuer is making in this segment — performance assertions, strategic positions, forward-looking statements. Empty list if the segment is purely descriptive.",
    )
    decisions: list[Decision] = Field(
        description="Decisions, appointments, approvals, or actions announced in the segment. Empty list if none.",
    )
    risks: list[Risk] = Field(
        description="Risks, uncertainties, or threats mentioned in the segment. Empty list if none.",
    )
    internal_contradictions: list[Contradiction] = Field(
        description="Statements in THIS segment that contradict each other. Empty list (the common case) if the segment is internally consistent. Do NOT compare with other segments.",
    )
    salient_quotes: list[str] = Field(
        description="2–6 short verbatim quotes that best capture the segment's substance. Useful for downstream global reasoning.",
    )
    notable_other: list[str] = Field(
        description="Anything materially important that didn't fit the other buckets (methodology notes, definitions, disclaimers, commitments). Each item is one short sentence; prefix with a category label when useful, e.g., 'definition: An active customer is one with...'. Empty list if none.",
    )


class AggregateAnalysis(BaseModel):
    """Aggregated analysis over multiple child analyses.

    Used for BOTH levels of reduction:
    - Section level: reduces N segment analyses (sharing the same `section_name`)
      into one section analysis.
    - Chapter level: reduces N section analyses (sharing the same `chapter_name`)
      into one chapter analysis.

    Same field set as `SegmentAnalysis` EXCEPT:
    - `internal_contradictions` is removed.
    - `contradictions` is added. This is the SINGLE contradiction sink at this
       level — it is an append-with-dedup list that absorbs every contradiction
       surfaced from children PLUS any new cross-element contradictions discovered
       during this reduction.
    """

    _decode_list_fields = field_validator(
        "topics",
        "key_entities",
        "metrics",
        "key_claims",
        "decisions",
        "risks",
        "contradictions",
        "salient_quotes",
        "notable_other",
        mode="before",
    )(_decode_list_field)

    ocr_confidence: Confidence = Field(
        description="Worst-case OCR confidence across all child analyses (use the lowest seen — if any child was 'low', this is 'low').",
    )
    has_table: bool = Field(
        description="True if ANY child analysis had has_table=true.",
    )
    has_figure: bool = Field(
        description="True if ANY child analysis had has_figure=true.",
    )
    summary: str = Field(
        description="A faithful 5–10 sentence synthesis that captures what the children collectively say. Cite key facts, named entities, and figures. No new information; do not invent. Maintain the narrative order of the children.",
    )
    topics: list[str] = Field(
        description="Deduplicated union of topic tags from the children. Merge near-duplicates into a single canonical form.",
    )
    sentiment: Sentiment = Field(
        description="Overall sentiment across the children: 'positive' / 'negative' / 'neutral' / 'mixed'. Use 'mixed' if children disagree substantively.",
    )
    key_entities: list[Entity] = Field(
        description="Deduplicated entities across children. Merge entities that refer to the same real-world thing (e.g., 'The Bank' and 'ICICI Bank' → one entry under the canonical name). Union the aliases. Combine roles/contexts into one descriptive sentence each.",
    )
    metrics: list[Metric] = Field(
        description="Union of metrics across children. If two children report the SAME metric for the SAME time_period, keep one entry and prefer the more specific value_text. Do not merge metrics with different time_periods.",
    )
    key_claims: list[Claim] = Field(
        description="Union of claims across children. Deduplicate near-identical claims (same polarity, same temporal_scope, same gist). Keep the supporting_quote from the child whose wording is more concrete.",
    )
    decisions: list[Decision] = Field(
        description="Union of decisions across children. Deduplicate identical decisions.",
    )
    risks: list[Risk] = Field(
        description="Union of risks across children. Deduplicate near-identical risks (same risk_category and same gist). If children disagree on severity, escalate (high beats medium beats low).",
    )
    contradictions: list[Contradiction] = Field(
        description=(
            "Append-with-dedup contradictions sink. Include: "
            "(1) every contradiction surfaced by children (their `internal_contradictions` if children are segments, or `contradictions` if children are sections); "
            "(2) every NEW contradiction you find ACROSS children at this reduction level — e.g., two segments in the same section that disagree on a number, an entity's role, or a forward-looking statement. "
            "Deduplicate: if two children flagged the same contradiction, keep one entry and union the supporting_quotes."
        ),
    )
    salient_quotes: list[str] = Field(
        description="3–10 short verbatim quotes selected from across the children that best capture the substance of this reduction. Drop redundant near-duplicates.",
    )
    notable_other: list[str] = Field(
        description="Union of notable_other strings across children, deduplicated by semantic equivalence (merge near-duplicates into one entry).",
    )
