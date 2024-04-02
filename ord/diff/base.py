from __future__ import annotations

from enum import Enum

from ord_schema import reaction_pb2

WORKUP_TYPES = list(reaction_pb2.ReactionWorkup.ReactionWorkupType.keys())


class MessageType(str, Enum):
    """ type of ord messages defined for message level comparison """

    REACTION_WORKUP = "REACTION_WORKUP"
    """ ord.ReactionWorkup """

    COMPOUND = "COMPOUND"
    """ ord.Compound """

    PRODUCT_COMPOUND = "PRODUCT_COMPOUND"
    """ ord.ProductCompound """

    REACTION_CONDITIONS = "REACTION_CONDITIONS"
    """ ord.ReactionConditions """

    REACTION = "REACTION"
    """ ord.Reaction """


class DeltaType(str, Enum):
    """ used to describe a result entry from `deepdiff` """

    ADDITION = "ADDITION"
    """ ref is None, act is not None (addition) """

    REMOVAL = "REMOVAL"
    """ ref is not None, act is None (removal) """

    ALTERATION = "ALTERATION"
    """ ref is not None, act is not None (alteration) """


class CompoundLeafType(str, Enum):
    """ they should be *disjoint* so any leaf of a compound can only be one of the these classes """

    reaction_role = 'reaction_role'
    """ an enum leaf for roles """

    identifiers = 'identifiers'
    """ all leafs under ord.Compound.identifiers or ord.ProductCompound.identifiers e.g., ord.CompoundIdentifier """

    amount = 'amount'
    """ all leafs under amount """

    other = 'other'
    """ all other fields """
