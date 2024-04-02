from __future__ import annotations

import json
import re
from collections import defaultdict

from loguru import logger
from ord_schema import reaction_pb2
from ord_schema.message_helpers import build_compound
from ord_schema.units import UnitResolver
from pandas._typing import FilePath

from ord.data.cre_data import CrePassage, CreLabel, CreReaction
from ord.data.data_reaction import ReactionData

"""
1. the set of passage_id in prod.txt is a proper superset of that in role.txt, 
this follows the definitions of these two tasks
2. a segment in role.txt is the same as a sentence in prod.txt, 
i.e. role[passage_id][segment_id] == prod[passage_id][sentence_id], not sure why it has two names
3. prod[passage_id].keys().sort() is arithmetic with delta=1
4. role[passage_id].keys() is a subset of prod[passage_id].keys()

For LLM, we have the choice to use passage (from prod.txt) or subpassage (from role.txt) as the prompt.
The latter should be an easier task.
"""

_SINGULAR_ROLES = ("Yield", "Temperature", "Time", "Reaction", "Prod", "Solvent")


def is_arithmetic(seq: list[int | float]):
    """ if the sequence is an arithmetic sequence"""
    gen = (i - j for i, j in zip(seq[:-1], seq[1:]))
    return all(d == d0 for d in gen for d0 in gen)


def to_sections(filename: FilePath, delimiter: str = "\n\n"):
    """ split a file to sections """
    with open(filename, "r") as f:
        text = f.read()
    pattern = re.compile(delimiter)
    sections = [s.strip() for s in pattern.split(text)]
    return sections


def parse_prod_section(sec: str, sentence_term="sentence="):
    """ parse one section in the `prod.txt` file """
    if not len(sec.strip()):
        return
    lines = sec.split("\n")
    lines = [li.strip() for li in lines if len(li.strip())]
    assert len(lines)
    passage_line = lines[0]
    content_lines = lines[1:]
    assert passage_line.startswith("#\tpassage")
    _, passage_id, sentence_id = passage_line.strip().split("\t")
    sentence_id = sentence_id[len(sentence_term):]
    sentence_id = int(sentence_id)

    sentence = ""
    for line in content_lines:
        items = line.strip().split("\t")
        token = items[0]
        sentence += token
        sentence += " "
    sentence = sentence.strip()
    return passage_id, sentence_id, sentence


def parse_prod(filename: FilePath = "prod.txt"):
    sections = to_sections(filename, "\n\n")
    sentence_dict = defaultdict(dict)
    for sec in sections:
        res = parse_prod_section(sec)
        if res is None:
            continue
        passage_id, sentence_id, sentence = res
        sentence_dict[passage_id][sentence_id] = sentence
    return sentence_dict


def reaction_from_ann_token_list(
        reaction_index: int,
        passage_id: str,
        sentence_id: int,
        annotated_token_list: list[tuple[str, CreLabel]],
        singular_roles: tuple[str, ...] = _SINGULAR_ROLES
) -> CreReaction | None:
    i = 0
    entities = []
    while i < len(annotated_token_list):
        token, label = annotated_token_list[i]
        is_end = i == len(annotated_token_list) - 1
        if label.type == "B":
            new_entity = token
            if not is_end:
                j = 1
                while i + j < len(annotated_token_list) and annotated_token_list[i + j][1].type == "I":
                    assert annotated_token_list[i + j][1].role == label.role
                    new_entity += annotated_token_list[i + j][0]
                    j += 1
            entities.append((label.role, new_entity))
        i += 1
    role_entities = defaultdict(list)

    for role, entity in entities:
        role_entities[role].append(entity)

    # curation decision: ignore reaction where multiple entities are found for a singular field
    role_to_unique_entities = dict()
    for role, entities in role_entities.items():
        is_singular_role = role in singular_roles
        unique_entities = sorted(set(entities))
        if is_singular_role and len(unique_entities) > 1:
            logger.warning(
                f"ignore reaction index: {reaction_index} as multiple entities is found for the singular role: {role}, the entity dictionary is: {role_entities}")
            return
        if len(unique_entities) == 0:
            unique_entities = ""
        elif len(unique_entities) == 1 and is_singular_role:
            unique_entities = unique_entities[0]
        role_to_unique_entities[role] = unique_entities
    reaction = CreReaction(passage_id=passage_id, sentence_id=sentence_id, reaction_index=reaction_index,
                           **dict(role_to_unique_entities))
    return reaction


def parse_role_section(
        sec: str, passage_id: str, sentence_id: int,
        singular_roles: tuple[str, ...] = _SINGULAR_ROLES
):
    if not len(sec.strip()):
        return
    lines = sec.split("\n")
    lines = [li.strip() for li in lines if len(li.strip())]
    assert len(lines)
    content_lines = lines[1:]
    n_prod = None
    for line in content_lines:
        items = line.strip().split("\t")
        n_prod_this_line = len(items) - 1
        assert n_prod is None or n_prod == n_prod_this_line
        n_prod = n_prod_this_line

    reactions = []
    for reaction_index in range(n_prod):
        annotated_token_list = []

        for line in content_lines:
            items = line.strip().split("\t")
            token = items[0]
            label = items[reaction_index + 1]
            assert label in list(CreLabel)
            entity = (token, CreLabel(label))
            annotated_token_list.append(entity)
        reaction = reaction_from_ann_token_list(reaction_index, passage_id, sentence_id, annotated_token_list,
                                                singular_roles)
        if reaction is not None:
            reactions.append(reaction)
    return reactions


def parse_role(
        filename: FilePath = "role.txt",
        singular_roles: tuple[str, ...] = _SINGULAR_ROLES
) -> list[CreReaction]:
    sections = to_sections(filename, "\n\n")
    reactions = []
    for sec in sections:
        res1 = parse_prod_section(sec, sentence_term="segment=")
        if res1 is None:
            continue
        passage_id, sentence_id, _ = res1
        res2 = parse_role_section(sec, passage_id, sentence_id, singular_roles)
        if res2 is None:
            continue
        reactions += res2
    return reactions


def collect_passages(
        prod_file: FilePath = "prod.txt",
        role_file: FilePath = "role.txt",
        use_subpassage: bool = False,
        only_one_reaction: bool = True,
        singular_roles: tuple[str, ...] = _SINGULAR_ROLES,
):
    """
    collect passages from prod.txt and role.txt

    :param prod_file:
    :param role_file:
    :param use_subpassage: if construct passage text from a subset of the sentences
    :param only_one_reaction: if only collect passages containing only one reaction
    :param singular_roles:
    :return:
    """
    sentence_dict = parse_prod(prod_file)
    reactions = parse_role(role_file, singular_roles)
    passage_id_to_reactions = defaultdict(list)

    for r in reactions:
        passage_id_to_reactions[r.passage_id].append(r)

    passages = []
    for passage_id in sentence_dict:
        text = ""
        if use_subpassage:
            sentence_ids = [r.sentence_id for r in passage_id_to_reactions[passage_id]]
        else:
            sentence_ids = sentence_dict[passage_id]
        for sentence_id in sentence_ids:
            text += sentence_dict[passage_id][sentence_id]
        passage = CrePassage(
            passage_id=passage_id,
            passage_text=text,
            is_subpassage=use_subpassage,
            contains=passage_id_to_reactions[passage_id]
        )
        if only_one_reaction and len(passage['contains']) != 1:
            logger.warning(f"more than one reaction found for: {passage_id}, this passage is therefore skipped!")
            continue
        else:
            if len(passage['contains']) > 1:
                logger.warning(f"more than one reaction found for: {passage_id}")
            passages.append(passage)
    logger.info(f"collected passages: {len(passages)}")
    logger.info(f"collected reactions: {len([r for p in passages for r in p['contains']])}")
    return passages


def convert_cre_reaction_to_reaction_data(reaction: CreReaction, text: str) -> ReactionData:
    unit_resolver = UnitResolver()

    ord_reaction = reaction_pb2.Reaction()

    ord_reaction.notes.procedure_details = text
    ord_reaction_id = f"ord-CRE-{reaction.passage_id}-{reaction.sentence_id}-{reaction.reaction_index}"
    ord_reaction.reaction_id = ord_reaction_id

    reaction_role_dict = {
        "Reactants": "REACTANT",
        "Catalyst_Reagents": "CATALYST",
        "Solvent": "SOLVENT",
    }
    for k in reaction.model_fields_set:
        if k in reaction_role_dict:
            ord_role = reaction_role_dict[k]
            if isinstance(getattr(reaction, k), list):
                names = getattr(reaction, k)
            elif isinstance(getattr(reaction, k), str):
                names = [getattr(reaction, k), ]
            else:
                raise TypeError
            for name in names:
                ord_reaction.inputs[name].components.add().CopyFrom(
                    build_compound(name=name, role=ord_role)
                )

    if "Temperature" in reaction:
        temp_text = reaction['Temperature']
        if temp_text.lower() == "room":
            temp_text = "25 C"
        temp_text = temp_text.replace("“C", "C")
        temp_text = temp_text.replace(",", "")
        temp_text = temp_text.replace("OC", " C")
        temp_text = temp_text.replace("'C", " C")
        try:
            temp_message = unit_resolver.resolve(temp_text)
        except (ValueError, KeyError):
            logger.error(f"failed to resolve temperature text: {temp_text}")
            temp_message = None
        if temp_message is not None:
            t_conds = ord_reaction.conditions.temperature
            t_conds.setpoint.CopyFrom(temp_message)

    # only one product
    outcome = ord_reaction.outcomes.add()
    product = outcome.products.add()
    product.identifiers.add(type="NAME", value=reaction.Prod)
    product.reaction_role = reaction_pb2.ReactionRole.PRODUCT

    if "Time" in reaction:
        time_text = reaction["Time"]
        try:
            time_message = unit_resolver.resolve(time_text)
        except (ValueError, KeyError) as e:
            logger.error(f"failed to resolve time text: {time_text}")
            time_message = None
        if time_message is not None:
            outcome.reaction_time.CopyFrom(time_message)

    if "Yield" in reaction:
        yield_text = reaction["Yield"]
        try:
            assert "%" in yield_text
            yield_value = float(yield_text.replace("%", "").strip())
            product.measurements.add(type="YIELD", percentage=dict(value=yield_value))
        except (AssertionError, ValueError):
            logger.error(f"failed to resolve yield text: {yield_text}")

    return ReactionData.from_reaction_message(ord_reaction)


def passage_to_reaction_data_list(p: CrePassage) -> list[ReactionData]:
    reactions = p["contains"]
    passage_text = p['passage_text']
    ld_data = []
    for r in reactions:
        ld = convert_cre_reaction_to_reaction_data(r, text=passage_text)
        ld_data.append(ld)
    return ld_data


if __name__ == '__main__':
    for version in ["singular", "multi", "one2many"]:
        logger.remove()
        logger.add(__file__.replace(".py", f"-{version}.log"))

        if version == "singular":
            one_reaction = True
        else:
            one_reaction = False

        PASSAGES = collect_passages(
            use_subpassage=True,
            only_one_reaction=one_reaction,
            singular_roles=_SINGULAR_ROLES,
        )
        LD_DATA = []
        for PASSAGE in PASSAGES:
            LD_DATA += passage_to_reaction_data_list(PASSAGE)

        if version in ["singular", "multi"]:
            logger.info(f"# of reactions: {len(LD_DATA)}")
            with open(f"CRE_data_{version}.json", 'w') as output_fp:
                json.dump([ld.model_dump() for ld in LD_DATA], output_fp, indent=2)
        else:
            LD_DATA_one2many = dict()
            for LD_DATA_entry in LD_DATA:
                try:
                    LD_DATA_one2many[LD_DATA_entry.procedure_text].append(LD_DATA_entry.model_dump())
                except KeyError:
                    LD_DATA_one2many[LD_DATA_entry.procedure_text] = [LD_DATA_entry.model_dump()]

            with open(f"CRE_data_{version}.json", 'w') as output_fp:
                json.dump(LD_DATA_one2many, output_fp, indent=2)
