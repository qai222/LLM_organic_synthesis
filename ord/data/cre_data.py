from __future__ import annotations

from enum import Enum
from typing import TypedDict

from pydantic import BaseModel
from typing import Optional


class CreLabel(str, Enum):
    """ Labels used in ChemRxnExtractor """
    O = "O"
    B_Reactants = "B-Reactants"
    I_Reactants = "I-Reactants"
    B_Catalyst_Reagents = "B-Catalyst_Reagents"
    I_Catalyst_Reagents = "I-Catalyst_Reagents"
    B_Workup_reagents = "B-Workup_reagents"
    I_Workup_reagents = "I-Workup_reagents"
    B_Reaction = "B-Reaction"
    I_Reaction = "I-Reaction"
    B_Solvent = "B-Solvent"
    I_Solvent = "I-Solvent"
    B_Yield = "B-Yield"
    I_Yield = "I-Yield"
    B_Temperature = "B-Temperature"
    I_Temperature = "I-Temperature"
    B_Time = "B-Time"
    I_Time = "I-Time"
    B_Prod = "B-Prod"
    I_Prod = "I-Prod"

    @property
    def role(self):
        if self == CreLabel.O:
            return self
        return self[2:]

    @property
    def type(self):
        if self == CreLabel.O:
            return self
        return self[0]


class CreReaction(BaseModel):
    """ from https://github.com/jiangfeng1124/ChemRxnExtractor/blob/main/configs/role_labels.txt """

    reaction_index: int
    """ the same sentence can describe multiple reactions, this is the column index in role.txt """

    passage_id: str

    sentence_id: int

    Reactants: Optional[list[str]] = None
    Catalyst_Reagents: Optional[list[str]] = None
    Workup_reagents: Optional[list[str]] = None
    Prod: Optional[str] = None
    Reaction: Optional[str] = None
    Solvent: Optional[str]  = None
    Yield: Optional[str] = None
    Temperature: Optional[str] = None
    Time: Optional[str] = None


class CrePassage(TypedDict):
    passage_id: str
    passage_text: str
    is_subpassage: bool  # is this constructed from role.txt (True) or prod.txt (False)?
    contains: list[CreReaction]  # reaction indices
