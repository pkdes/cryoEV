"""
Shared list helper for the portal's list pages (round 13, feedback #22): filter, sort and group rows from a plain GET
form, with the state kept in the URL. Server-side only (no JS). Used by Results, Predict "My analyses" and the Reviews
hub (portal/modules/reviews.py).

INTERFACE CONTRACT (fixed by the PM before the build; B2 owns this file and fills the bodies; B1 and B2 call it):

    facets(ps_dir, request, *, viewer=None, hidden=None) -> dict
        Group/filter keys for one prediction set, from its request.json (`request`, already parsed) and folder:
          person   owner id (agent-made sets stay under their owner)
          via      owner_via or None (the "include agent-made" toggle)
          created  epoch seconds (float) or None
          session  request source.session, else "upload · YYYY-MM-DD"
          source   label of the root of the derived_from chain (prediction_set / dataset links only), else the set's
                   own display name
          model    request model id
          samples  list of sample labels from images.csv (may be empty)
        Every key that names another item passes through portal.modules.data.records.ref_hidden (via `hidden`,
        a callable(kind, id) -> bool for this viewer), so a private set id never shows as a group heading.

    parse_query(query_params) -> ListQuery
        Reads ?person=&agents=&from=&to=&model=&status=&q=&sort=&group= (unknown values fall back to defaults).
        Defaults: person any, agents on, no dates, model any, status "open" on the Reviews hub and "any" elsewhere
        (the caller passes default_status), q "", sort "newest", group "none".

    apply(rows, q, *, key=lambda r: r["facets"]) -> list
        Filter + sort rows (dicts). Each row's facets come from key(row); rows also carry "status" and "title".

    group(rows, by, *, key=lambda r: r["facets"]) -> list[tuple[str, list]]
        by in {"none","person","session","source","model","sample"}; "none" -> [("", rows)].
        sample: a row with <= 3 samples sits in each sample's group; more -> "Mixed (more than 3 samples)".

    controls_html(q, *, action, people, models, statuses, show_group=True) -> str
        The GET form (escaped), state prefilled from q. Each control has a stable id.

ListQuery is a frozen dataclass with the fields above plus .to_query() -> str for links.
"""
from __future__ import annotations

from dataclasses import dataclass

GROUPS = ("none", "person", "session", "source", "model", "sample")
SORTS = ("newest", "name", "size")
MIXED = "Mixed (more than 3 samples)"


@dataclass(frozen=True)
class ListQuery:
    person: str = "any"
    agents: bool = True
    date_from: str = ""
    date_to: str = ""
    model: str = "any"
    status: str = "any"
    q: str = ""
    sort: str = "newest"
    group: str = "none"

    def to_query(self) -> str:
        raise NotImplementedError("B2 fills this in")


def facets(ps_dir, request, *, viewer=None, hidden=None) -> dict:
    raise NotImplementedError("B2 fills this in")


def parse_query(query_params, *, default_status: str = "any") -> ListQuery:
    raise NotImplementedError("B2 fills this in")


def apply(rows, q: ListQuery, *, key=lambda r: r["facets"]) -> list:
    raise NotImplementedError("B2 fills this in")


def group(rows, by: str, *, key=lambda r: r["facets"]) -> list:
    raise NotImplementedError("B2 fills this in")


def controls_html(q: ListQuery, *, action: str, people, models, statuses, show_group: bool = True) -> str:
    raise NotImplementedError("B2 fills this in")
