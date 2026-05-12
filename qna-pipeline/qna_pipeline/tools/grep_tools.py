"""Direct grep + page-text tools exposed to the GRA orchestrator.

These are bounded-by-construction (snippet cap, match limit, token cap),
so the GRA can call them directly without going through a sub-agent.
"""

import json
import logging
from typing import Callable, Optional

from langchain_core.tools import tool

from config.settings import GREP_MATCH_LIMIT
from qna_pipeline.db.markdown import PageNotFound
from qna_pipeline.db.markdown import get_page_text as get_page_text_impl
from qna_pipeline.db.markdown import grep as grep_impl

log = logging.getLogger(__name__)


def _json_default(obj):
    try:
        return str(obj)
    except Exception:
        return None


def make_grep_tools(document_id: str) -> tuple[Callable, Callable]:
    """Build `(grep, get_page_text)` tools scoped to `document_id`.

    Both close over the doc_id so the inner ReAct loop doesn't need to
    propagate it through tool arguments.
    """

    @tool
    async def grep(
        pattern: str,
        pages_filter: Optional[list[int]] = None,
        regex: bool = False,
        limit: int = GREP_MATCH_LIMIT,
    ) -> str:
        """Case-insensitive search over the preprocessed markdown for this
        document. Returns up to `limit` matches.

        Use for: needle-in-haystack questions, exact-phrase lookups, "does
        the document mention X anywhere", "on which page is Y stated".

        Args:
            pattern: Text to find. Treated literally (regex-escaped) when
                regex=False (default).
            pages_filter: Optional list of PRINTED page numbers to restrict
                the search to. Pages with no footer mapping are silently
                skipped.
            regex: When True, treat `pattern` as a Python regex.
            limit: Max matches to return (capped at GREP_MATCH_LIMIT).

        Returns:
            JSON array of {printed_page, physical_page, line, snippet}.
            Empty array if no matches.
        """
        log.info(
            f"[grep:{document_id}] pattern={pattern!r} "
            f"pages_filter={pages_filter} regex={regex} limit={limit}"
        )
        matches = await grep_impl(
            document_id,
            pattern,
            pages_filter=pages_filter,
            regex=regex,
            limit=limit,
        )
        return json.dumps(matches, ensure_ascii=False, default=_json_default)

    @tool
    async def get_page_text(printed_pages: list[int]) -> str:
        """Return the raw markdown text for the requested PRINTED page
        numbers (the ones shown in the document's footer / stored in `pages`
        arrays).

        The output is truncated to fit a per-call token budget; if you
        request many pages or one page is unusually long, the tail will
        be cut and a notice appended.

        Use for: fetching context around a grep hit, reading a specific
        page the user named, or inspecting a page whose number you already
        know from a chapter / section aggregate.

        Args:
            printed_pages: List of printed page numbers to retrieve.

        Returns:
            The joined page text, with `--- Printed page N ---` separators.
            If NONE of the requested pages have a footer mapping (front
            matter / unnumbered pages), returns an error message — try a
            nearby printed page from a section/chapter `pages` array.
        """
        log.info(f"[get_page_text:{document_id}] printed_pages={printed_pages}")
        try:
            return await get_page_text_impl(document_id, printed_pages)
        except PageNotFound as e:
            return f"[error: {e}]"

    return grep, get_page_text
