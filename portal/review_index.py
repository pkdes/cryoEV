"""
One list of every review package a person may open (round 13, the Reviews hub). Data only, no routes: it lives
outside portal/modules/ because modules.discover() imports every module there.

INTERFACE CONTRACT (fixed by the PM before the build; B1 owns this file and fills the bodies; B3's switcher strip
and B1's hub call it):

    list_reviews(cfg, user, can_see, *, show_all=False, access_mode=False) -> list[dict]
        Every row is filtered through reviews.may_open BEFORE it is built, and built from this field allowlist only
        (never review_package.json "meta", "cards", "path", decoy counts, or who/via on audits):
          id          package folder name
          kind        "prediction" | "label_review" | "audit" | "sandbox"
          title       readable name
          made_from   short text ("v5 F5 · 677 images · Session 20261106-2")
          owner       portal id, or None on audits and the sandbox
          reviewers   list of portal ids ([] on audits)
          role        "yours" | "reviewer" | ""
          status      "not_started" | "in_progress" | "finished" | "applied"
          cards, decided   ints from reviews.card_progress, or None
          n_images    int or None
          last        epoch seconds of the last edit (else built time), or None
          open_url    "/review/<id>/", "/review/" (core package, admin only), the sandbox link, or None
          go_url      "/reviews/go/<id>"
          mode        "cards" if cards are open, else "editor"
          needs_you   bool: owner or reviewer, status not_started/in_progress, open_url set, not the owner of an audit
          can_assign  bool: owner or admin, a served pred_/lab_ package, never the core package
          blind       bool (audits)
          via         owner_via or None (None on audits)
          facets      portal.listing.facets(...) for prediction rows, else {} (B2's helper)
        Excluded/core duplicates are dropped (the core row opens /review/); dot-folders and folders the multi server
        can't serve are skipped; the sandbox is admin only and absent when access_mode is True.

    needs_you(rows) -> list[dict]
        The "Needs your review" rows, most recent activity first.

    switcher_groups(rows, current_id, *, per_group=8) -> list[tuple[str, list[dict]]]
        Groups for the switcher strip's dropdown: "Needs you" first, then by kind, at most per_group rows each.
"""
from __future__ import annotations

KINDS = ("prediction", "label_review", "audit", "sandbox")
STATUSES = ("not_started", "in_progress", "finished", "applied")


def list_reviews(cfg, user, can_see, *, show_all=False, access_mode=False) -> list:
    raise NotImplementedError("B1 fills this in")


def needs_you(rows) -> list:
    raise NotImplementedError("B1 fills this in")


def switcher_groups(rows, current_id, *, per_group: int = 8) -> list:
    raise NotImplementedError("B1 fills this in")
