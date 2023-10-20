i  # further improved version

import string

import numpy as np
from ord_schema import message_helpers
from ord_schema.proto import reaction_pb2, dataset_pb2
from ord_schema.units import UnitResolver


# lit = LineIterator(filename="data/role/test_subset.txt")
# with open("data/role/test_subset.txt", "r", newline="") as f:
#     lines = f.readlines()


# step 1: parse section data into a dictionary
def helper_parse_section(file_name):
    with open(file_name, "r", newline="") as f:
        lines = f.readlines()

    # a dictionary to store the section data
    # doi: list of lines
    section_data = []

    for line in lines:
        # comment line with dio
        if line.strip().startswith("#"):
            # create a sub-record dictionary as the beginning of a new subsection
            doi = line.strip().split()[1].lstrip("passage=")
            subsection_data = {"doi": doi, "words": ""}
        # line break because of white empty line
        elif line.strip() == "":
            # assign the record values as this subsection ends here
            section_data.append(subsection_data)
        # actual parsing goes here
        else:
            # word = line.split()[0]
            # tokens = line.split()[1:]
            subsection_data["words"] += line

    # append the last subsection data in case the last line is not a line break
    if line != "\n":
        section_data.append(subsection_data)

    return section_data


# step 2: parse the section data into a record dictionary
def helper_parse_rxn_roles(subsection_data, role_tag, product_idx=0, return_text=True):
    """Parse reaction roles for each product in a record.

    Parameters
    ----------
    subsection_data : str
        The text section containing the reaction roles.
    role_tag : str
        The tag for the reaction role.
    return_text : bool, optional
        Whether to return the text for the reaction role, by default False.

    Returns
    -------
    dict
        A dictionary of reaction roles.

    """
    role_record = []
    lines = subsection_data["words"].split("\n")
    line = lines[0]
    # number of products
    num_products = len(line.split()) - 1

    for line in lines:
        if line != "":
            tokens = line.split()
            word = tokens[0]
            label = tokens[product_idx + 1]
            if ("B-" + role_tag) == label:
                # parse the reaction roles
                role_record.append(word)
            elif ("I-" + role_tag) == label:
                role_record[-1] += " " + word
            else:
                continue
        else:
            continue

    text_list = []
    for line in lines:
        if line != "":
            text_list.append(line.split()[0])
    # plain_text = " ".join(text_list)
    # plain_text = "".join(
    #     [word if word in string.punctuation else " " + word for word in text_list]
    # )
    plain_text = text_list[0] + "".join(
        [
            # word if (word in string.punctuation and "(" not in word) else " " + word
            word if word in string.punctuation.replace("(", "") else " " + word
            for word in text_list[1:]
        ]
    )
    plain_text = plain_text.replace("- ", "-")

    # role_record = None if len(role_record) == 0 else role_record[0]
    # if len(role_record) == 0:
    #     role_record = None
    # else:
    #     role_record = role_record[0]

    # fix text representation problems from the original text
    if role_tag == "Temperature":
        role_record = [temperature.replace("° C", "°C") if temperature is not None else None
                       for temperature in role_record]
    # fix extra spaces in yield
    if role_tag == "Yield":
        role_record = [yield_prod.replace(" %", "%") if yield_prod is not None else None
                       for yield_prod in role_record]

    if return_text:
        return role_record, plain_text
    else:
        return role_record


def helper_get_num_products(subsection_data):
    """Get the number of products in a record."""
    lines = subsection_data["words"].split("\n")
    num_products = len(lines[0].split()) - 1

    return num_products


# step 3: a wrapper function to parse the whole file
def parse_rxn_roles(file_name):
    """Parse reaction roles for each product in the plain text file."""
    section_data = helper_parse_section(file_name)

    record_list = []
    for subsection_data in section_data:
        # number of products
        num_products = helper_get_num_products(subsection_data)
        # parse the reaction roles
        # https://github.com/FanwangM/ChemRxnExtractor/blob/main/configs/role_labels.txt

        reactants = []
        products = []
        solvents = []
        yields_prod = []
        catalyst_reagents = []
        reaction_time_list = []
        workup_reagents = []
        temperature_list = []
        for i in range(num_products):
            reactant = helper_parse_rxn_roles(
                subsection_data, "Reactants", product_idx=i, return_text=False)
            product, plain_text = helper_parse_rxn_roles(
                subsection_data, "Prod", product_idx=i, return_text=True)
            solvent = helper_parse_rxn_roles(
                subsection_data, "Solvent", product_idx=i, return_text=False)
            yield_prod = helper_parse_rxn_roles(
                subsection_data, "Yield", product_idx=i, return_text=False)
            reaction_time = helper_parse_rxn_roles(
                subsection_data, "Time", product_idx=i, return_text=False
            )
            catalyst_reagent = helper_parse_rxn_roles(
                subsection_data, "Catalyst_Reagent", product_idx=i, return_text=False
            )
            workup_reagent = helper_parse_rxn_roles(
                subsection_data, "Workup_Reagent", product_idx=i, return_text=False
            )
            temperature = helper_parse_rxn_roles(
                subsection_data, "Temperature", product_idx=i, return_text=False
            )

            reactants.append(reactant)
            products.append(product)
            solvents.append(solvent)
            catalyst_reagents.append(catalyst_reagent)
            yields_prod.append(yield_prod)
            workup_reagents.append(workup_reagent)
            reaction_time_list.append(reaction_time)
            temperature_list.append(temperature)

        # # fix text representation problems from the original text
        # if len(temperature_list) != 0:
        #     temperature_list = [temperature.replace("° C", "°C") for temperature in
        #                         temperature_list if temperature is not None]
        #
        # # fix extra spaces in yield
        # if len(yields_prod) != 0:
        #     yields_prod = [
        #         yield_prod.replace(" %", "%") if yield_prod is not None else None
        #         for yield_prod in yields_prod
        #     ]

        # fix hyphens and spaces
        plain_text = plain_text.replace("- ", "-")
        # fix temperature unit representation
        plain_text = plain_text.replace("° C", "°C")
        # fix left parenthesis and spaces
        plain_text = plain_text.replace("( ", "(")

        # create a record dictionary
        record = {
            "doi": subsection_data["doi"],
            "plain_text": plain_text,
            "products": products,
            "reactants": reactants,
            "catalyst_reagents": catalyst_reagents,
            "workup_reagents": workup_reagents,
            # "reaction": reaction,
            "solvent": solvents,
            "yield": yields_prod,
            "temperature": temperature_list,
            "time": reaction_time_list,
        }

        record_list.append(record)

    return record_list


# step 4: parse the parsed data into a ORD dataframe
def build_ord_database(record_list, output_file=None):
    """Build ORD database."""
    unit_resolver = UnitResolver()

    reactions_ord_list = []
    # reaction.provenance.doi = doi

    for record in record_list:
        # get number of reactions in each record
        num_reactions = len(record["products"])
        # get shared information
        plain_text = record["plain_text"]
        doi = record["doi"]

        for idx in np.arange(num_reactions):
            # create a new reaction record
            reaction = reaction_pb2.Reaction()
            # provenance
            # reaction.provenance.city = "Cambridge, USA"

            # reaction.provenance.doi = doi
            reaction.provenance.doi = doi.split("-")[0]
            # reaction.provenance.record_created.time.value = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            # reaction.provenance.record_created.person.CopyFrom(
            #     reaction_pb2.Person(name="Fanwang Meng",
            #                         organization="MIT",
            #                         email="fwmeng@mit.edu"))

            # assigning values to the reaction record
            # plain text
            reaction.notes.procedure_details = record["plain_text"]

            # reactants
            if len(record["reactants"][idx]) != 0:
                # reactant_ord = reaction.inputs["reactants"]
                # for reactant in record["reactants"][idx]:
                #     reactant_ord.identifiers.add(type="NAME", value=reactant)
                for reactant in record["reactants"][idx]:
                    reactant_ord = reaction.inputs["reactants"].components.add()
                    reactant_ord.reaction_role = reaction_pb2.ReactionRole.REACTANT
                    reactant_ord.identifiers.add(type="NAME", value=reactant)

            # workup reagents
            if len(record["workup_reagents"][idx]) != 0:
                input_workup = reaction.inputs[record["workup_reagents"]]
                # solute = input_sf.components.add()
                workup_reagent_ord = input_workup.components.add()
                workup_reagent_ord.reaction_role = reaction_pb2.ReactionRole.REAGENT
                for workup_reagent in record["workup_reagents"][idx]:
                    workup_reagent_ord.identifiers.add(type="NAME", value=workup_reagent)

            # temperature
            if len(record["temperature"][idx]) != 0:
                try:
                    t_conds = reaction.conditions.temperature
                    t_conds.setpoint.CopyFrom(unit_resolver.resolve(record["temperature"][idx][0]))
                except:
                    print(f"Error found for parsing the temperature information for {record['doi']}")
                    continue

            # products and their yields
            outcome = reaction.outcomes.add()

            for idx, pro in enumerate(record["products"][idx]):
                if len(record["time"][idx]) != 0:
                    try:
                        outcome.reaction_time.CopyFrom(unit_resolver.resolve(record["time"][idx][0]))
                    except:
                        print(f"Error found for parsing the reaction time information for {record['doi']}")
                        continue
                product_ord = outcome.products.add()
                product_ord.identifiers.add(type="NAME", value=pro)

                if len(record["yield"][idx]) != 0:
                    try:
                        percent = dict(value=int(record["yield"][idx][0].strip("%")), precision=2)
                        product_ord.measurements.add(type="YIELD",
                                                     percentage=percent)
                    except:
                        print(f"Error found for parsing the yield information for {record['doi']}")
                        continue

            # catalysts
            if len(record["catalyst_reagents"][idx]) != 0:
                input_catalyst = reaction.inputs[record["catalyst_reagents"]]
                # solute = input_sf.components.add()
                catalyst_ord = input_catalyst.components.add()
                catalyst_ord.reaction_role = reaction_pb2.ReactionRole.CATALYST
                for catalyst in record["catalyst_reagents"][idx]:
                    catalyst_ord.identifiers.add(type="NAME", value=catalyst)

            # solvent
            if len(record["solvent"][idx]) != 0:
                input_sf = reaction.inputs["solvent"]
                # solute = input_sf.components.add()
                solute = input_sf.components.add()
                solute.reaction_role = reaction_pb2.ReactionRole.SOLVENT
                for solvent in record["solvent"][idx]:
                    solute.identifiers.add(type="NAME", value=solvent)

            reactions_ord_list.append(reaction)

    # validate the reactions
    # this result into errors because of required information not available
    # for reaction in reactions_ord_list:
    #     try:
    #         # validate the reaction
    #         validations.validate_message(reaction)
    #         # add the reaction to the dataset
    #         # dataset.reactions.add().CopyFrom(reaction)
    #     except:
    #         print(f"Error found for {reaction.provenance.doi}")

    # create a dataset
    dataset = dataset_pb2.Dataset(
        name="Dataset from ChemRxnExtractor paper",
        description="Reaction data from J.Chem.Inf.Model.2022,62,2035−2045 (https://doi.org/10.1021/acs.jcim.1c00284)",
        reactions=reactions_ord_list)

    if output_file is not None:
        message_helpers.write_message(dataset, output_file)
        print(f"Congratulations! ORD database is built successfully and saved to {output_file}.")

    return reactions_ord_list, dataset


if __name__ == "__main__":
    input_file = "train_dev_test.txt"
    output_file = "train_dev_test.pb"

    record_list = parse_rxn_roles(input_file)
    _, _ = reactions_ord_list, dataset = build_ord_database(record_list,
                                                            output_file=output_file)
