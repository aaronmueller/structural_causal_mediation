"""Run all the extraction for a model across many templates.
"""
import argparse
import os
from datetime import datetime

import torch
from transformers import GPT2Tokenizer

from experiment import Intervention, Model, StrTemplates
from utils import convert_results_to_pd

from grammar import read_grammar

parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")

parser.add_argument(
    "-model",
    type=str,
    default="distilgpt2",
    help="""Model type [distilgpt2, gpt-2, etc.].""",
)

parser.add_argument(
    "-out_dir", default=".", type=str, help="""Path of the result folder."""
)

parser.add_argument(
    "-template_indices",
    nargs="+",
    type=int,
    help="Give the indices of templates if you want to run on only a subset",
)

parser.add_argument(
    "--randomize", default=False, action="store_true", help="Randomize model weights."
)

parser.add_argument(
    "--grammar_file", type=str, default=None
)

parser.add_argument(
    "--structure", type=str, default=None
)

opt = parser.parse_args()


def get_profession_list(grammar=None, structure=None):
    # Get the list of all considered professions
    if grammar is None:
        word_list = []
        with open("experiment_data/professions.json", "r") as f:
            for l in f:
                # there is only one line that eval"s to an array
                for j in eval(l):
                    word_list.append(j[0])
        return word_list
    else:
        if structure.startswith("across") or structure == "simple_agreement":
            word_list = {'sing': grammar[("N1", frozenset("s"))],
                         'pl':   grammar[("N1", frozenset("p"))]}
        elif structure.startswith("within"):
            word_list = {'sing': grammar[("N2", frozenset("s"))],
                         'pl':   grammar[("N2", frozenset("p"))]}
        return word_list


def get_template_list(indices=None, structure=None, grammar=None):
    # Get list of all considered templates
    # "That" sentences are ours
    # "Because" sentences are a subset
    # from https://arxiv.org/pdf/1807.11714.pdf (Lu et al.)
    if structure is None:
        templates = [
            "The {} said that",
            "The {} yelled that",
            "The {} whispered that",
            "The {} wanted that",
            "The {} desired that",
            "The {} wished that",
            "The {} ate because",
            "The {} ran because",
            "The {} drove because",
            "The {} slept because",
            "The {} cried because",
            "The {} laughed because",
            "The {} went home because",
            "The {} stayed up because",
            "The {} was fired because",
            "The {} was promoted because",
            "The {} yelled because",
        ]
        if indices:
            subset_templates = [templates[i - 1] for i in indices]
            print("subset of templates:", subset_templates)
            return subset_templates
        return templates

    elif structure == "simple_agreement":
        templates = StrTemplates("The {}", structure, grammar)
    elif structure == "across_obj_rel":
        templates = StrTemplates("The {} that the {} {}", structure, grammar)
    elif structure == "across_subj_rel":
        templates = StrTemplates("The {} that {} the {}", structure, grammar)
    elif structure == "within_obj_rel":
        templates = StrTemplates("The {} that the {}", structure, grammar)

    return templates.base_strings


def get_intervention_types(bias_type=None):
    if bias_type is None:
        return [
            "man_direct",
            "man_indirect",
            "woman_direct",
            "woman_indirect",
        ]
    elif bias_type == "structural":
        return [
            "diffnum_direct",
            "diffnum_indirect"
        ]


def construct_interventions(base_sent, professions, tokenizer, DEVICE, structure=None, number=None,
                            subs=None):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    if structure is None:
        for p in professions:
            all_word_count += 1
            try:
                interventions[p] = Intervention(
                    tokenizer, base_sent, [p, "man", "woman"], ["he", "she"], device=DEVICE
                )
                used_word_count += 1
            except:
                pass
        return
    # else we're doing structural interventions
    if structure.startswith("across") or structure == "simple_agreement":
        candidate_sing = "is"; candidate_pl = "are"
    elif structure.startswith("within"):
        candidate_sing = "likes"; candidate_pl = "like"
    if structure == "across_subj_rel": 
        sub = base_sent.split()[-1]
    elif structure == "across_obj_rel":
        sub = base_sent.split()[-2]
    elif structure == "within_obj_rel":
        sub = base_sent.split()[1]
    for idx, p in enumerate(professions):
        all_word_count += 1
        if structure == "simple_agreement":
            sub = subs[idx]
        # print(base_sent, p, sub, candidate_pl, candidate_sing, DEVICE)
        try:
            if number == "sing":
                interventions[p] = Intervention(
                    tokenizer, base_sent, [p, sub], [candidate_pl, candidate_sing], device=DEVICE,
                    structure=structure
                )
            elif number == "pl":
                interventions[p] = Intervention(
                    tokenizer, base_sent, [p, sub], [candidate_sing, candidate_pl], device=DEVICE,
                    structure=structure
                )
            used_word_count += 1
        except:
            pass

    print(
        "\t Only used {}/{} professions due to tokenizer".format(
            used_word_count, all_word_count
        )
    )
    return interventions


def run_all(
    model_type="gpt2",
    device="cuda",
    out_dir=".",
    grammar_file=None,
    structure=None,
    random_weights=False,
    template_indices=None,
):
    print("Model:", model_type, flush=True)
    # Set up all the potential combinations.
    if grammar_file is not None:
        if structure is None:
            raise Exception("Error: grammar file given but no structure specified")
        grammar = read_grammar(grammar_file)
        professions = get_profession_list(grammar=grammar, structure=structure)
        templates = get_template_list(structure=structure, grammar=grammar)
        intervention_types = get_intervention_types(bias_type="structural")
    else:
        professions = get_profession_list()
        templates = get_template_list(template_indices)
        intervention_types = get_intervention_types()

    # Initialize Model and Tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = Model(device=device, gpt2_version=model_type, random_weights=random_weights)

    # Set up folder if it does not exist.
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string + "_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Iterate over all possible templates.
    for number in ('sing', 'pl'):
        for temp in templates[number]:
            print("Running template '{}' now...".format(temp), flush=True)
            # Fill in all professions into current template
            if structure == "simple_agreement":
                other_number = 'sing' if number == 'pl' else 'pl'
                interventions = construct_interventions(temp, professions[number], tokenizer, device, structure=structure, number=number, subs=professions[other_number])
            else:
                interventions = construct_interventions(temp, professions[number], tokenizer, device, structure=structure, number=number)
            # Consider all the intervention types
            for itype in intervention_types:
                print("\t Running with intervention: {}".format(itype), flush=True)
                # Run actual exp.
                intervention_results = model.neuron_intervention_experiment(
                    interventions, itype, alpha=1.0
                )

                df = convert_results_to_pd(interventions, intervention_results)
                # Generate file name.
                temp_string = "_".join(temp.replace("{}", "X").split())
                model_type_string = model_type
                fname = "_".join([temp_string, itype, model_type_string])
                # Finally, save each exp separately.
                df.to_csv(os.path.join(base_path, fname + ".csv"))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    run_all(
        opt.model,
        device,
        opt.out_dir,
        opt.grammar_file,
        opt.structure,
        random_weights=opt.randomize,
        template_indices=opt.template_indices
    )
