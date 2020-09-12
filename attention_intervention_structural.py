"""Performs attention intervention on Winobias samples and saves results to JSON file."""

import json

import fire
from pandas import DataFrame
from transformers import GPT2Tokenizer

import winobias
from attention_utils import perform_interventions, get_odds_ratio
from experiment import Model, Intervention, StrTemplates

from grammar import read_grammar


def get_interventions_winobias(gpt2_version, do_filter, split, model, tokenizer,
                                device='cuda', filter_quantile=0.25):
    if split == 'dev':
        examples = winobias.load_dev_examples()
    elif split == 'test':
        examples = winobias.load_test_examples()
    else:
        raise ValueError(f"Invalid split: {split}")
    json_data = {'model_version': gpt2_version,
            'do_filter': do_filter,
            'split': split,
            'num_examples_loaded': len(examples)}
    if do_filter:
        interventions = [ex.to_intervention(tokenizer) for ex in examples]
        df = DataFrame({'odds_ratio': [get_odds_ratio(intervention, model) for intervention in interventions]})
        df_expected = df[df.odds_ratio > 1]
        threshold = df_expected.odds_ratio.quantile(filter_quantile)
        filtered_examples = []
        assert len(examples) == len(df)
        for i in range(len(examples)):
            ex = examples[i]
            odds_ratio = df.iloc[i].odds_ratio
            if odds_ratio > threshold:
                filtered_examples.append(ex)

        print(f'Num examples with odds ratio > 1: {len(df_expected)} / {len(examples)}')
        print(
            f'Num examples with odds ratio > {threshold:.4f} ({filter_quantile} quantile): {len(filtered_examples)} / {len(examples)}')
        json_data['num_examples_aligned'] = len(df_expected)
        json_data['filter_quantile'] = filter_quantile
        json_data['threshold'] = threshold
        examples = filtered_examples
    json_data['num_examples_analyzed'] = len(examples)
    interventions = [ex.to_intervention(tokenizer) for ex in examples]
    return interventions, json_data

def load_structural_interventions(tokenizer, device, structure=None):
    grammar = read_grammar('structural/grammar.avg')
    if structure.startswith("across") or structure == "simple_agreement":
        professions = {'sing': grammar[("N1", frozenset("s"))],
                       'pl':   grammar[("N1", frozenset("p"))]}
    elif structure.startswith("within"):
        professions = {'sing': grammar[("N2", frozenset("s"))],
                       'pl':   grammar[("N2", frozenset("p"))]}
    if structure == "simple_agreement":
        templates = StrTemplates("The {}", structure, grammar)
    elif structure == "across_obj_rel":
        templates = StrTemplates("The {} that the {} {}", structure, grammar)
    elif structure == "across_subj_rel":
        templates = StrTemplates("The {} that {} the {}", structure, grammar)
    elif structure == "within_obj_rel":
        templates = StrTemplates("The {} that the {}", structure, grammar)
    templates = templates.base_strings
    intervention_types = ["diffnum_direct", "diffnum_indirect"]

    # build list of interventions
    interventions = []
    if structure.startswith("across") or structure == "simple_agreement":
        candidate_sing = "is"; candidate_pl = "are"
    elif structure.startswith("within"):
        candidate_sing = "likes"; candidate_pl = "like"

    for number in ('sing', 'pl'):
        if structure == "simple_agreement":
            other_number = "sing" if number == "pl" else "pl"
        for template in templates[number]:
            if structure.startswith("within"):
                sub = template.split()[-1]
            elif structure == "across_obj_rel":
                sub = template.split()[-2]
            elif structure == "across_subj_rel":
                sub = template.split()[-1]
            elif structure == "within_obj_rel":
                sub = template.split()[1]
            for idx, p in enumerate(professions[number]):
                if structure == "simple_agreement":
                    sub = professions[other_number][idx]
                try:
                    if number == "sing":
                        interventions.append(Intervention(
                            tokenizer, template, [p, sub], 
                            [candidate_sing, candidate_pl],
                            device=device, structure=structure
                        ))
                    elif number == "pl":
                        interventions.append(Intervention(
                            tokenizer, template, [p, sub],
                            [candidate_pl, candidate_sing],
                            device=device, structure=structure
                        ))
                except:
                    pass

    return interventions


def get_interventions_structural(gpt2_version, do_filter, model, tokenizer,
                                 device='cuda', filter_quantile=0.25, structure=None):
    interventions = load_structural_interventions(tokenizer, device, structure=structure)
    
    json_data = {'model_version': gpt2_version,
            'do_filter': do_filter,
            'num_examples_loaded': len(interventions)}
    if do_filter:
        df = DataFrame({'odds_ratio': [get_odds_ratio(intervention, model) for intervention in interventions]})
        df_expected = df[df.odds_ratio > 1]
        threshold = df_expected.odds_ratio.quantile(filter_quantile)
        filtered_interventions = []
        assert len(interventions) == len(df)
        for i in range(len(interventions)):
            intervention = interventions[i]
            odds_ratio = df.iloc[i].odds_ratio
            if odds_ratio > threshold:
                filtered_interventions.append(intervention)

        print(f'Num examples with odds ratio > 1: {len(df_expected)} / {len(interventions)}')
        print(
            f'Num examples with odds ratio > {threshold:.4f} ({filter_quantile} quantile): {len(filtered_interventions)} / {len(interventions)}')
        json_data['num_examples_aligned'] = len(df_expected)
        json_data['filter_quantile'] = filter_quantile
        json_data['threshold'] = threshold
        interventions = filtered_interventions
    json_data['num_examples_analyzed'] = len(interventions)
    return interventions, json_data


def intervene_attention(gpt2_version, do_filter, structure, device='cuda', filter_quantile=0.25, random_weights=False):
    model = Model(output_attentions=True, gpt2_version=gpt2_version, device=device, random_weights=random_weights)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)

    interventions, json_data = get_interventions_structural(gpt2_version, do_filter,
                                                            model, tokenizer,
                                                            device, filter_quantile,
                                                            structure=structure)
    results = perform_interventions(interventions, model)
    json_data['mean_total_effect'] = DataFrame(results).total_effect.mean()
    json_data['mean_model_indirect_effect'] = DataFrame(results).indirect_effect_model.mean()
    json_data['mean_model_direct_effect'] = DataFrame(results).direct_effect_model.mean()
    filter_name = 'filtered' if do_filter else 'unfiltered'
    if random_weights:
        gpt2_version += '_random'
    fname = f"structural_attention/{structure}/attention_intervention_{gpt2_version}_{filter_name}.json"
    json_data['results'] = results
    with open(fname, 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    fire.Fire(intervene_attention)
