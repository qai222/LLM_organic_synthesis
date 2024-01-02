from __future__ import annotations

from enum import Enum

from ord_schema import reaction_pb2

WORKUP_TYPES = list(reaction_pb2.ReactionWorkup.ReactionWorkupType.keys())


class MessageType(str, Enum):
    REACTION_WORKUP = "REACTION_WORKUP"
    COMPOUND = "COMPOUND"
    PRODUCT_COMPOUND = "PRODUCT_COMPOUND"
    REACTION_CONDITIONS = "REACTION_CONDITIONS"
    REACTION = "REACTION"


class DeltaType(str, Enum):
    """ used to describe a result entry from `deepdiff` """

    # ref is None, act is not None (addition)
    ADDITION = "ADDITION"

    # ref is not None, act is None (removal)
    REMOVAL = "REMOVAL"

    # ref is not None, act is not None (alteration)
    ALTERATION = "ALTERATION"


class CompoundLeafType(str, Enum):
    """ they should be *disjoint* so any leaf of a compound can only be one of the these classes """

    reaction_role = 'reactionRole'
    # NOTE: this would have been `reaction_role`
    # but field names are turned into camelCase in protobuf serializer by default

    identifiers = 'identifiers'

    amount = 'amount'

    other = 'other'
