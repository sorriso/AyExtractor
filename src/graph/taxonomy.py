# src/graph/taxonomy.py — v1
"""Default relation taxonomy constants.

Contains the base taxonomy table from spec §13.4.
See spec §13.4 for taxonomy structure and extension rules.
"""

from __future__ import annotations

from ayextractor.core.models import RelationTaxonomyEntry

DEFAULT_RELATION_TAXONOMY: list[RelationTaxonomyEntry] = [
    # Hierarchical
    RelationTaxonomyEntry(canonical_relation="is_a", original_forms=["is_a", "is a", "est un", "est une"], category="hierarchical", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="part_of", original_forms=["part_of", "is part of", "fait partie de", "belongs to"], category="hierarchical", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="contains", original_forms=["contains", "contient", "includes", "inclut"], category="hierarchical", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="subclass_of", original_forms=["subclass_of", "subclass of", "sous-classe de", "type of"], category="hierarchical", is_directional=True),
    # Composition
    RelationTaxonomyEntry(canonical_relation="has_component", original_forms=["has_component", "has component", "a comme composant"], category="composition", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="composed_of", original_forms=["composed_of", "composed of", "composé de"], category="composition", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="member_of", original_forms=["member_of", "member of", "membre de"], category="composition", is_directional=True),
    # Causality
    RelationTaxonomyEntry(canonical_relation="causes", original_forms=["causes", "cause", "provoque", "entraîne"], category="causality", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="enables", original_forms=["enables", "permet", "facilitates"], category="causality", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="prevents", original_forms=["prevents", "empêche", "blocks"], category="causality", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="leads_to", original_forms=["leads_to", "leads to", "mène à", "conduit à"], category="causality", is_directional=True),
    # Governance
    RelationTaxonomyEntry(canonical_relation="regulates", original_forms=["regulates", "réglemente", "governs", "encadre"], category="governance", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="requires", original_forms=["requires", "exige", "nécessite", "mandates"], category="governance", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="complies_with", original_forms=["complies_with", "complies with", "se conforme à", "conforme à"], category="governance", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="enforces", original_forms=["enforces", "applique", "impose"], category="governance", is_directional=True),
    # Temporal
    RelationTaxonomyEntry(canonical_relation="precedes", original_forms=["precedes", "précède", "before"], category="temporal", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="follows", original_forms=["follows", "suit", "after"], category="temporal", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="concurrent_with", original_forms=["concurrent_with", "concurrent with", "simultané avec"], category="temporal", is_directional=False),
    # Attribution
    RelationTaxonomyEntry(canonical_relation="created_by", original_forms=["created_by", "created by", "créé par"], category="attribution", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="authored_by", original_forms=["authored_by", "authored by", "rédigé par", "écrit par"], category="attribution", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="owned_by", original_forms=["owned_by", "owned by", "détenu par", "appartient à"], category="attribution", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="employs", original_forms=["employs", "emploie", "hires"], category="attribution", is_directional=True),
    # Location
    RelationTaxonomyEntry(canonical_relation="located_in", original_forms=["located_in", "located in", "situé dans", "basé à"], category="location", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="applies_to", original_forms=["applies_to", "applies to", "s'applique à", "concerne"], category="location", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="operates_in", original_forms=["operates_in", "operates in", "opère dans"], category="location", is_directional=True),
    # Association
    RelationTaxonomyEntry(canonical_relation="related_to", original_forms=["related_to", "related to", "lié à", "associé à"], category="association", is_directional=False),
    RelationTaxonomyEntry(canonical_relation="similar_to", original_forms=["similar_to", "similar to", "similaire à"], category="association", is_directional=False),
    RelationTaxonomyEntry(canonical_relation="contrasts_with", original_forms=["contrasts_with", "contrasts with", "contraste avec", "s'oppose à"], category="association", is_directional=False),
    RelationTaxonomyEntry(canonical_relation="references", original_forms=["references", "référence", "cite", "mentionne"], category="association", is_directional=True),
    # Production
    RelationTaxonomyEntry(canonical_relation="produces", original_forms=["produces", "produit", "generates", "génère"], category="production", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="generates", original_forms=["generates", "génère", "yields"], category="production", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="implements", original_forms=["implements", "implémente", "met en œuvre"], category="production", is_directional=True),
    RelationTaxonomyEntry(canonical_relation="defines", original_forms=["defines", "définit", "specifies", "spécifie"], category="production", is_directional=True),
]


def get_categories() -> list[str]:
    """Return distinct taxonomy categories."""
    return sorted({e.category for e in DEFAULT_RELATION_TAXONOMY})


def find_canonical(raw_relation: str) -> str | None:
    """Find canonical relation for a raw form (exact match).

    Returns None if no match found.
    """
    normalized = raw_relation.lower().strip()
    for entry in DEFAULT_RELATION_TAXONOMY:
        if normalized in (f.lower() for f in entry.original_forms):
            return entry.canonical_relation
        if normalized == entry.canonical_relation:
            return entry.canonical_relation
    return None


def get_taxonomy_dict() -> dict[str, RelationTaxonomyEntry]:
    """Return taxonomy indexed by canonical relation name."""
    return {e.canonical_relation: e for e in DEFAULT_RELATION_TAXONOMY}
