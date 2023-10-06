from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, List
import omegaconf
import pandas as pd
from plot import AccuracyPlot, Heatmap
import matplotlib.pyplot as plt
from process import process_experiment_runs
from omegaconf import OmegaConf
from utils import *

class Experiment:
    def __init__(self, params: dict, base_dir: str, params_for_label: List[str], params_name: dict) -> None:
        self.params = params
        self.dir = os.path.join(base_dir, params['dataset'], self.name)
        self.params_for_label = params_for_label
        self.params_name = params_name

    @property
    def label(self) -> str:
        params = {}
        for key in self.params:
            if key in self.params_for_label:
                params[key] = get_param_name_mapping(self.params_name, key, self.params[key])

        return ' '.join(params.values())

    @property
    def name(self) -> str:
        return '_'.join(self.params.values())
    
    def has_same_params(self, params: dict) -> bool:
        return all(self.params[k] == val for k, val in params.items())
    
    def __repr__(self) -> str:
        return self.name
    

def get_exp(params: omegaconf.DictConfig, *args, **kwargs) -> List[Experiment]:
    params_product = get_dict_product(params)
    return [Experiment(p, *args, **kwargs) for p in params_product]

def gen_param_groups(exp_config: omegaconf.DictConfig) -> List[Dict[str, str]]:
    """Generate groups of parameters corresponding to the product of parameters values.
       Used to group together experiments in the same figure based on having the same
       combination of some parameters (e.g. group together experiments on the same dataset)

    Args:
        exp_config (omegaconf.DictConfig): configuration of the global experiment to plot

    Returns:
        List[Dict[str, str]]: dictionaries each corresponding to a single configuration of parameters
                              (e.g. [{dataset -> cifar10, optim -> sgdm}, {dataset -> cifar100, optim -> sgdm}])
    """
    groupby_values = {k: set() for k in exp_config.groupby}
    for p in exp_config.params:
        for k in exp_config.groupby:
            groupby_values[k].update(p[k])
    return get_dict_product(groupby_values)

def gen_experiments(exp_config: omegaconf.DictConfig, exp_groups: List[Dict[str, str]], 
                    params_name_map: omegaconf.DictConfig) -> Dict[str, List[Experiment]]:
    """Generate experiments from the global experiment specification and groups them according to the configurations
       in exp_groups

    Args:
        exp_config (omegaconf.DictConfig): global experiment specification
        exp_groups (List[Dict[str, str]]): dictionaries each corresponding to a single configuration of parameters
        params_name_map (omegaconf.DictConfig): mapping for parameters values (e.g. {algo -> {fedavg -> FedAvg},...}, ...)

    Returns:
        Dict[str, List[Experiment]]: experiments specified in exp_config, grouped according exp_groups
    """
    experiments = []
    for p in exp_config.params:
        experiments.extend(get_exp(p, exp_config.dir, exp_config.params_for_label, params_name_map))
    experiments = {get_name_from_values(plot_p): list(filter(lambda e: e.has_same_params(plot_p), experiments))
                                                 for plot_p in exp_groups}
    return experiments

def extract_plot_prefs(global_pref: omegaconf.DictConfig, groupby_values: List[str]) -> omegaconf.DictConfig:
    """Obtain the plot preferences for the experiments belongign to a group

    Args:
        global_pref (omegaconf.DictConfig): mapping for values of groping params to plot preferences 
                                            (e.g. resnet.cifar10 -> {...prefs})
        groupby_values (List[str]): list of values corresponding to params defining groups of experiments 
                                    (e.g. [lenet, cifar100], which correspond to group keys [model, dataset])

    Returns:
        omegaconf.DictConfig: preferences for the experimets group
    """
    pref_keys = list(global_pref)
    global_pref_set = [set(pref.split('.')) for pref in pref_keys]
    index, _ = find_longest_subset(global_pref_set, groupby_values)
    if index is not None:
        return global_pref[pref_keys[index]]
    return global_pref['default']

def accuracy_plot(exps: dict, plots: dict, plot_pref: dict, mappings: dict, summary: pd.DataFrame) -> pd.DataFrame:
    for group_name in exps:
        group_experiments = exps[group_name]
        group_plot = plots[group_name]
        for e in group_experiments:
            try:
                exp_dir, label = e.dir, e.label
                line, df = process_experiment_runs(exp_dir, **plot_pref.process_config)
                summary = pd.concat([df, summary])
                light = color_light(mappings.color_light, e.params)
                color = adjust_lightness(mappings.algo_color.get(e.params['algo'], None), light)
                group_plot.addline(label, line, **plot_pref.lines_config, color=color)
            except Exception as ex:
                print(f"Failed to process experiment {e.name}: {ex}")
        group_plot.showlegend()
    return summary

def final_accuracy_plot(exps: dict, plots: dict, plot_pref: dict, mappings: dict, summary: pd.DataFrame):
    for group_name in exps:
        group_experiments = exps[group_name]
        group_plot = plots[group_name]
        data = defaultdict(dict)
        colors = {}
        for e in group_experiments:
            try:
                exp_dir, label, name = e.dir, e.label, e.name
                plot_over_value = float(e.params[plot_pref.plot_over].removeprefix(plot_pref.plot_over))
                if name in summary.index: 
                    data[label][plot_over_value] = summary.loc[e.name][plot_pref.plot_column]
                else:
                    _, df = process_experiment_runs(exp_dir)
                    summary = pd.concat([df, summary])
                    data[label][plot_over_value] = df.loc[name][plot_pref.plot_column]
                light = color_light(mappings.color_light, e.params)
                color = adjust_lightness(mappings.algo_color[e.params['algo']], light)
                colors[label] = color
            except Exception as ex:
                print(f"Failed to process experiment {e.name}: {ex}")
        for label, val in data.items():
            color = colors[label]
            group_plot.addline(label, {'accuracy': val}, **plot_pref.lines_config, color=color)
        group_plot.showlegend()
    return summary

def savefigs(figs: Dict[str, AccuracyPlot], path: str):
    for f in figs.values():
        f.savefig(path)

def gen_final_accuracy_plot(title: str, groupby_values: List[dict], name_mappings: dict, plot, experiments, summary):
    mapped_values = [{name: get_param_name_mapping(name_mappings.params_name, name, value) for name, value in e.items()} for e in groupby_values]
    acc_plots = {get_name_from_values(k): AccuracyPlot(title, get_name_from_values(e, ' - '), **extract_plot_prefs(plot.plot_pref, list(k.values()))) 
                 for k, e in zip(groupby_values, mapped_values)}
    aggr_data = final_accuracy_plot(experiments, acc_plots, plot, name_mappings, summary)
    return acc_plots, aggr_data

def gen_accuracy_plot(title: str, groupby_values: List[dict], name_mappings: dict, plot, experiments, summary):
    mapped_values = [{name: get_param_name_mapping(name_mappings.params_name, name, value) for name, value in e.items()} for e in groupby_values]
    acc_plots = {get_name_from_values(k): AccuracyPlot(title, get_name_from_values(e, ' - '), **extract_plot_prefs(plot.plot_pref, list(k.values()))) 
                 for k, e in zip(groupby_values, mapped_values)}
    aggr_data = accuracy_plot(experiments, acc_plots, plot, name_mappings, summary)
    return acc_plots, aggr_data

def heatmap_plot(exps: dict, plots: dict, plot_pref: dict, mappings: dict, summary: pd.DataFrame):
    for group_name in exps:
        group_experiments = exps[group_name]
        group_plot = plots[group_name]
        x_key, y_key = plot_pref.params_for_label.x, plot_pref.params_for_label.y
        for e in group_experiments:
            try:
                exp_dir = e.dir
                _, df = process_experiment_runs(exp_dir, **plot_pref.process_config)
                summary = pd.concat([df, summary])
                row_name = get_param_name_mapping(mappings.params_name, x_key, e.params[x_key])
                col_name = get_param_name_mapping(mappings.params_name, y_key, e.params[y_key])
                group_plot.addline(row_name, col_name, df.iloc[0,0], df.iloc[0,1])
            except Exception as ex:
                print(f"Failed to process experiment {e.name}: {ex}")
        group_plot.showlegend()
    return summary

def gen_heatmaps(title: str, groupby_values: List[dict], name_mappings: dict, plot, experiments, summary) -> pd.DataFrame:
    # Generate plots
    mapped_values = [{name: get_param_name_mapping(name_mappings.params_name, name, value) for name, value in e.items()} for e in groupby_values]
    x_values = ordered_values(plot.params_for_label.x, values_set(plot.params_for_label.x, plot.params), name_mappings.params_name)
    y_values = ordered_values(plot.params_for_label.y, values_set(plot.params_for_label.y, plot.params), name_mappings.params_name)
    acc_plots = {get_name_from_values(k): Heatmap(title, get_name_from_values(e, ' - '), True, 18,
                                                   row_labels=x_values, col_labels=y_values,
                                                  **extract_plot_prefs(plot.plot_pref, list(k.values()))) 
                 for k, e in zip(groupby_values, mapped_values)}
    aggr_data = heatmap_plot(experiments, acc_plots, plot, name_mappings, summary)
    return acc_plots, aggr_data


def process_plots(plots: dict , name_map: dict, summary: pd.DataFrame):
    for plot in plots:
        param_groups = gen_param_groups(plot)
        experiments = gen_experiments(plot, param_groups, name_map.params_name)
        plot_fn = eval(plot.function)
        figs, aggregated_data = plot_fn(plot.title, param_groups, name_map, plot, experiments, summary)
        if plot.savefig_path:
            plot_path = os.path.join(plot.savefig_path, plot.title)
            os.makedirs(plot_path, exist_ok=True)
            savefigs(figs, plot_path)
        summary = pd.concat([summary, aggregated_data])
        summary = summary.groupby(level=0).last()
    summary.sort_index(inplace=True)
    return summary

def main(args):
    summary_path = Path(args.summary_file)
    summary = retrieve_summary(summary_path)
    # Process plots
    summary = process_plots(args.plots, args.mappings, summary)
    summary.to_csv(summary_path)
    print("Summary", summary, sep='\n')

if __name__ == '__main__':
    register_omegaconf_resolvers()
    args = OmegaConf.load('config.yaml')
    plt.rcParams.update(args.plt_config)
    main(args)
    plt.show()