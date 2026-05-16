"""
reputation_memory.py

Per-agent structured memory for tracking Player 0's evolving observations of peers
in the Avalon game. Updated once per round via an LLM call; injected into the prompt
context before each discussion / action phase.

Components
----------
SemanticMemory  : synthesised belief about a peer (label, confidence, justification)
Interactions    : append-only lists of observed alliances and conflicts
PeerRecord      : container pairing the two components for a single peer
ReputationMemory: top-level dict of PeerRecord objects, one per peer
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SemanticMemory:
    """Synthesised belief about a single peer.

    Attributes
    ----------
    alignment_role:
        Free-form string label, e.g. "Unknown", "Trustworthy", "Likely Evil",
        "Likely Merlin", "Confirmed Evil".  Not an Enum so the LLM can express
        nuance without brittle parsing.
    confidence_score:
        Integer in [1, 5].  1 = highly uncertain, 5 = absolutely certain.
        Overwritten each round (later observations supersede earlier ones).
    justification:
        One concise sentence grounding the belief in an observable game event.
        Overwritten each round.
    """
    alignment_role:   str = "Unknown"
    confidence_score: int = 1
    justification:    str = "No observations yet."

    def __post_init__(self) -> None:
        self.confidence_score = max(1, min(5, int(self.confidence_score)))


@dataclass
class Interactions:
    """Append-only lists of observed cooperative and antagonistic behaviours.

    Both lists grow over the course of the game; old entries are never removed.
    This ensures the LLM retains the full behavioural history of a peer.

    Attributes
    ----------
    alliances:
        E.g. ["Always votes Accept on teams proposed by Player C",
               "Defended Player B in Round 2"].
    conflicts:
        E.g. ["Repeatedly votes Reject on Player D's proposals",
               "Accused Player E of being a Minion"].
    """
    alliances: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PeerRecord — one per observed peer
# ---------------------------------------------------------------------------

@dataclass
class PeerRecord:
    """Holds all reputation data for a single peer player.

    Attributes
    ----------
    is_locked:
        If True, this peer's semantic belief was set from ground-truth role
        knowledge at game start. The semantic fields will never be overwritten
        by LLM updates; only the Interactions lists (alliances/conflicts) may
        still be appended to.
    """
    semantic:     SemanticMemory = field(default_factory=SemanticMemory)
    interactions: Interactions   = field(default_factory=Interactions)
    is_locked:    bool           = False


# ---------------------------------------------------------------------------
# ReputationMemory — top-level container
# ---------------------------------------------------------------------------

class ReputationMemory:
    """Player 0's private, evolving memory of all peer players.

    Parameters
    ----------
    player_id:
        The id of the agent that *owns* this memory (i.e. Player 0).
    num_players:
        Total number of players in the game.

    Usage
    -----
    Initialise once at game start::

        memory = ReputationMemory(player_id=0, num_players=5)

    Update once per round (called by task.py)::

        await agent.update_reputation_memory(round_summary)

    Inject into the prompt before each action::

        agent.inject_reputation_context(label="team_discussion")
    """

    def __init__(self, player_id: int, num_players: int) -> None:
        self.player_id   = player_id
        self.num_players = num_players
        # One PeerRecord per peer (everyone except self)
        self.peers: Dict[int, PeerRecord] = {
            pid: PeerRecord()
            for pid in range(num_players)
            if pid != player_id
        }

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def lock_peer(
        self,
        peer_id: int,
        alignment_role: str,
        justification: str = "Role revealed at game start via faction knowledge.",
    ) -> None:
        """Pin a peer's semantic belief to ground-truth; immune to LLM overwriting.

        Only the Interactions (alliances / conflicts) lists may still grow.
        """
        if peer_id == self.player_id or peer_id not in self.peers:
            return
        pr = self.peers[peer_id]
        pr.semantic.alignment_role   = alignment_role
        pr.semantic.confidence_score = 5
        pr.semantic.justification    = justification
        pr.is_locked = True

    def locked_peer_ids(self) -> List[int]:
        """Return sorted list of peer ids whose beliefs are locked."""
        return sorted(pid for pid, pr in self.peers.items() if pr.is_locked)

    def update_from_llm(self, peer_id: int, record: dict) -> None:
        """Apply a single parsed LLM update record to a peer.

        Parameters
        ----------
        peer_id:
            The player id being updated.
        record:
            Dict with keys: alignment_role, confidence_score, justification,
            new_alliances (list[str]), new_conflicts (list[str]).

        Notes
        -----
        If the peer is locked (ground-truth knowledge from role reveals), the
        semantic fields (alignment_role, confidence_score, justification) are
        NOT overwritten. Interaction observations are still appended.
        """
        if peer_id == self.player_id or peer_id not in self.peers:
            return

        pr = self.peers[peer_id]

        if not pr.is_locked:
            # Overwrite semantic belief (later observations supersede earlier ones)
            pr.semantic.alignment_role   = str(record.get("alignment_role",   pr.semantic.alignment_role))
            pr.semantic.confidence_score = max(1, min(5, int(record.get("confidence_score", pr.semantic.confidence_score))))
            pr.semantic.justification    = str(record.get("justification",    pr.semantic.justification))

        # Interaction observations always append regardless of lock status
        new_alliances = record.get("new_alliances", [])
        new_conflicts = record.get("new_conflicts", [])
        if isinstance(new_alliances, list):
            pr.interactions.alliances.extend(new_alliances)
        if isinstance(new_conflicts, list):
            pr.interactions.conflicts.extend(new_conflicts)

    def apply_llm_updates(self, updates: list) -> None:
        """Apply a list of LLM update records (one per peer).

        Parameters
        ----------
        updates:
            Parsed JSON array returned by the LLM, each element matching the
            schema expected by :meth:`update_from_llm`.
        """
        for record in updates:
            try:
                peer_id = int(record.get("player_id", -1))
                self.update_from_llm(peer_id, record)
            except (ValueError, TypeError, KeyError):
                # Malformed record — skip silently, log at call site
                continue

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_prompt_block(self) -> str:
        """Render the full memory as a human-readable block for LLM injection.

        Locked peers are clearly flagged [CONFIRMED — do not update] so the
        LLM knows not to waste reasoning on them.
        """
        lines: List[str] = []
        for peer_id in sorted(self.peers):
            pr  = self.peers[peer_id]
            sem = pr.semantic
            itr = pr.interactions

            lock_tag = " [CONFIRMED — do not update]" if pr.is_locked else ""
            lines.append(f"Player {peer_id}:")
            lines.append(f"  Belief     : {sem.alignment_role} (confidence {sem.confidence_score}/5){lock_tag}")
            lines.append(f"  Reasoning  : {sem.justification}")

            if itr.alliances:
                lines.append("  Alliances  :")
                for obs in itr.alliances:
                    lines.append(f"    - {obs}")
            else:
                lines.append("  Alliances  : (none observed)")

            if itr.conflicts:
                lines.append("  Conflicts  :")
                for obs in itr.conflicts:
                    lines.append(f"    - {obs}")
            else:
                lines.append("  Conflicts  : (none observed)")

            lines.append("")  # blank separator between peers

        return "\n".join(lines).rstrip()

    def has_non_trivial_data(self) -> bool:
        """Return True if at least one peer has been updated beyond defaults."""
        for pr in self.peers.values():
            if pr.semantic.alignment_role != "Unknown" or pr.semantic.confidence_score > 1:
                return True
            if pr.interactions.alliances or pr.interactions.conflicts:
                return True
        return False

    # ------------------------------------------------------------------
    # Serialisation helpers (for logging / debugging)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            str(pid): {
                "semantic": {
                    "alignment_role":   pr.semantic.alignment_role,
                    "confidence_score": pr.semantic.confidence_score,
                    "justification":    pr.semantic.justification,
                },
                "interactions": {
                    "alliances": pr.interactions.alliances,
                    "conflicts": pr.interactions.conflicts,
                },
                "is_locked": pr.is_locked,
            }
            for pid, pr in self.peers.items()
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"ReputationMemory(player={self.player_id}, peers={list(self.peers.keys())})"
