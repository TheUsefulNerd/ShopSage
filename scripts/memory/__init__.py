#!/usr/bin/env python3
"""Cross-session user preference memory backed by Supabase.

Architecture:
  - Long-term (this module): structured preferences stored in Supabase
    user_preferences table. Survives server restarts and redeployments.
    Keyed by session_id (Streamlit's st.session_state gets a UUID on first load).
  - Short-term: in-session Streamlit st.session_state — known_products,
    conversation history, etc. Lives only for the browser session duration.

What is stored:
  - max_budget / min_budget: last explicitly stated budget constraint.
  - preferred_categories: list of category names the user has searched for.
  - session_count: how many sessions this user has had (for welcome message).
  - last_seen: timestamp of last visit.

Usage:
    from scripts.memory.store import MemoryStore
    store = MemoryStore(session_id="some-uuid")
    prefs = store.load()          # -> dict or None if first visit
    store.update(max_budget=150)  # persists immediately
    store.increment_session()     # call once per new session
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TABLE = "user_preferences"


def _get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise EnvironmentError("Missing SUPABASE_URL and/or SUPABASE_KEY.")
    return create_client(url, key)


class MemoryStore:
    """Supabase-backed preference store for a single session/user.

    Args:
        session_id: A stable identifier for this user/session. In Streamlit,
            generate a UUID once and store it in st.session_state["user_id"].
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._client: Optional[Client] = None

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = _get_supabase_client()
        return self._client

    def load(self) -> Optional[dict]:
        """Load preferences for this session_id. Returns None if first visit."""
        try:
            resp = (
                self._get_client()
                .table(TABLE)
                .select("*")
                .eq("session_id", self.session_id)
                .execute()
            )
            if resp.data:
                return resp.data[0]
            return None
        except Exception as e:
            print(f"[MemoryStore] load failed: {e}")
            return None

    def update(
        self,
        max_budget: Optional[float] = None,
        min_budget: Optional[float] = None,
        preferred_categories: Optional[list[str]] = None,
    ) -> bool:
        """Upsert preferences. Only provided fields are written; others unchanged.

        Returns True on success, False on failure (non-fatal — app keeps working).
        """
        existing = self.load()
        now = datetime.now(timezone.utc).isoformat()

        if existing is None:
            # First visit — insert new row
            row = {
                "session_id": self.session_id,
                "max_budget": max_budget,
                "min_budget": min_budget,
                "preferred_categories": preferred_categories or [],
                "last_seen": now,
                "session_count": 1,
            }
            try:
                self._get_client().table(TABLE).insert(row).execute()
                return True
            except Exception as e:
                print(f"[MemoryStore] insert failed: {e}")
                return False
        else:
            # Update only the fields that were provided
            updates: dict = {"last_seen": now}
            if max_budget is not None:
                updates["max_budget"] = max_budget
            if min_budget is not None:
                updates["min_budget"] = min_budget
            if preferred_categories is not None:
                # Merge with existing, dedup
                existing_cats = existing.get("preferred_categories") or []
                merged = list(dict.fromkeys(existing_cats + preferred_categories))
                updates["preferred_categories"] = merged
            try:
                (
                    self._get_client()
                    .table(TABLE)
                    .update(updates)
                    .eq("session_id", self.session_id)
                    .execute()
                )
                return True
            except Exception as e:
                print(f"[MemoryStore] update failed: {e}")
                return False

    def increment_session(self) -> None:
        """Increment session_count. Call once at the start of each new session."""
        try:
            existing = self.load()
            if existing is None:
                return  # update() will handle insert on first preference write
            new_count = (existing.get("session_count") or 1) + 1
            (
                self._get_client()
                .table(TABLE)
                .update({"session_count": new_count})
                .eq("session_id", self.session_id)
                .execute()
            )
        except Exception as e:
            print(f"[MemoryStore] increment_session failed: {e}")

    def build_greeting(self) -> Optional[str]:
        """Return a welcome-back message if this is a returning user, else None."""
        prefs = self.load()
        if prefs is None or prefs.get("session_count", 1) <= 1:
            return None

        parts = []
        if prefs.get("max_budget") is not None:
            parts.append(f"budget under ${prefs['max_budget']:.0f}")
        cats = prefs.get("preferred_categories") or []
        if cats:
            parts.append(f"interest in {', '.join(cats[:2])}")

        if parts:
            pref_text = " and ".join(parts)
            return (
                f"Welcome back! I remember your {pref_text}. "
                f"I'll keep that in mind — feel free to start a new search!"
            )
        return "Welcome back! What are you shopping for today?"

    def extract_and_save(self, intent) -> None:
        """Extract preferences from a UserIntent and persist them.

        Call this after every successful new_search so preferences are kept
        up to date without requiring the user to re-state them each session.

        Args:
            intent: A UserIntent (or any object with max_budget, min_budget,
                    search_query attributes).
        """
        max_b = getattr(intent, "max_budget", None)
        min_b = getattr(intent, "min_budget", None)
        query = getattr(intent, "search_query", "") or ""

        # Infer preferred category from search_query keywords
        category_keywords = {
            "Outdoor": ["tent", "camping", "outdoor", "shelter"],
            "Hiking": ["hiking", "trail", "trek", "backpack", "boots", "headlamp"],
            "Clothing": ["jacket", "hoodie", "jeans", "shirt", "sweater", "coat"],
            "Footwear": ["shoes", "boots", "sandals", "sneakers", "loafers", "footwear"],
            "Beauty": ["lipstick", "foundation", "serum", "powder", "cosmetics", "beauty"],
        }
        detected_cats = []
        query_lower = query.lower()
        for cat, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_cats.append(cat)

        if max_b is not None or min_b is not None or detected_cats:
            self.update(
                max_budget=max_b,
                min_budget=min_b,
                preferred_categories=detected_cats or None,
            )
