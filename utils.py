import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from itertools import product, combinations
import omegaconf

def search_in_list(l: list, val: Any) -> Optional[int]:
    """Analogous to index() method of list, but returns None instead of throwing exception

    Args:
        l (list): list to search in
        val (Any): value to search

    Returns:
        Optional[int]: the index of the element if found, None otherwise
    """
    for i, v in enumerate(l):
        if v == val:
            return i
    return None

def find_longest_subset(sets: List[Set[str]], values: List[str]) -> Tuple[Optional[int], Optional[Set[str]]]:
    """Finds the maximal k-subset from sets composed by k-values in values

    Args:
        sets (List[Set[str]]): list of sets
        values (List[str]): values for subsets

    Returns:
        Tuple[Optional[int], Optional[Set[str]]]: index of the subset and the subset itself if found, tuple of None otherwise
    """
    for l in range(len(values), 0, -1):
        for c in combinations(values, l):
            subset= set(c)
            i =  search_in_list(sets, subset) # sets.index(set(c))
            if i is not None:
                return i, subset
    return None, None

def replace_chars(text: str, excluding_chars: str, replacing_char: str) -> str:
    """Replaces all the occurrences of any characters in excluding chars with replacing_char in text

    Args:
        text (str): string to modify
        excluding_chars (str): chars to exclude from text
        replacing_char (str): char to substitute any occurence of characters in excluding_chars with

    Returns:
        str: the modified string
    """
    assert len(replacing_char) <= 1, "replacing_char is meant to be one char only"
    ex_char_set = set(excluding_chars)
    return ''.join([s if s not in ex_char_set else replacing_char for s in text])

def retrieve_summary(file_path: Path):
    if file_path.exists():
        return pd.read_csv(file_path, index_col=[0])
    return pd.DataFrame()

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    if color is None:
        return None
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def color_light(light_mapping: omegaconf.DictConfig, exp_params: Dict[str, str]) -> float:
    """Compute the color light corresponding to the parameters of current experiment

    Args:
        light_mapping (omegaconf.DictConfig): mapping from param name to mapping param_value -> color light
        exp_params (Dict[str, str]): parameters of current experiment (e.g. {algo -> fedavg, dataset -> cifar10, ...})

    Returns:
        float: color light for the current experiment
    """
    for key in light_mapping:
        if key in exp_params:
            return light_mapping[key].get(exp_params[key], 1)
    return 1.
    
def get_dict_product(di: Dict[str, Set[str]]) -> List[Dict[str, str]]:
    """Compute the product of all values corresponding to dictionary keys and returns them as list of dicts

    Args:
        di (Dict[str, Set[str]]): input dictionary mapping keys to a set of values 
                                  (e.g. {dataset -> {cifar10, cifar100}, optim -> {sgdm} })

    Returns:
        List[Dict[str, str]]: product dictionaries mapping each key to one value
                              (e.g. [{dataset -> cifar10, optim -> sgdm}, {dataset -> cifar100, optim -> sgdm}])
    """
    return [{k: v for k,v in zip(di.keys(), d)} for d in product(*di.values())]

def get_name_from_values(d: dict, sep: str = '.') -> str:
    return sep.join(d.values())

def get_param_name_mapping(mappings: omegaconf.DictConfig, param_name: str, value: str) -> str:
    """Maps a value for an experiment parameter to a string

    Args:
        mappings (omegaconf.DictConfig): mapping table for name conversion of all params
        param_name (str): name of the parameter, used to retrieve the correct mapping table
        value (str): value to map

    Returns:
        str: the mapping corresponding to value in the table
    """
    d = mappings.get(param_name, {})
    v = d.get(value, value)
    return v

def ordered_values(prefix: str, values: Set[str], mappings: omegaconf.DictConfig) -> List[str]:
    """Remap values and return them ordered in a list

    Args:
        prefix (str): the prefix common to all values, e.g. param name (tau)
        values (Set[str]): values to map, in any order, e.g. [tau1, tau4, tau2]
        mappings (omegaconf.DictConfig): dictionary-like mapping for prefix, e.g. tau2 -> $\tau=2$

    Returns:
        List[str]: list of mapped and ordered values, e.g. ['$\tau1$', '$\tau2$']
    """
    sorted_values = sorted(values, key=lambda x: x.removeprefix(prefix).zfill(3))
    map_fn = lambda v: get_param_name_mapping(mappings, prefix, v)
    return list(map(map_fn, sorted_values))

def values_set(param_name: str, param_groups: List[Dict[str, list]]) -> set:
    """Obtain the set of all values associated to a experiment parameter


    Args:
        param_name (str): the name of experiment parameter (e.g. bs)
        params (List[Dict[str, list]]): experiment params, consisting of dicts mapping parameters name to values
                                        (e.g. params[0] = {bs -> [bs16, bs32, bs64], optim -> [sgdm, adamw]})

    Returns:
        set: all the values associated with param_name for all given param groups 
    """
    param_values = []
    for p in param_groups:
        param_values.extend(p[param_name]) 
    return set(param_values)

class dictLikeMapper:
    """Helper class to generate a dict-like mapper based on key manipulation
    """
    def __init__(self, key: str, keyMapped: str, mode: str = 'normal') -> None:
        self.key = key
        self.keyMapped = keyMapped
        self.mode = mode
    def __getitem__(self, value: str) -> str:
        value = value.removeprefix(self.key)
        if self.mode == 'scientific':
            value = "{:.1e}".format(float(value))
        return f"${self.keyMapped}={value}$"
    def get(self, key, _=None):
        return self.__getitem__(key)
    
def register_omegaconf_resolvers():
    omegaconf.OmegaConf.register_new_resolver('mapper', lambda *x: dictLikeMapper(*x))
    omegaconf.OmegaConf.register_new_resolver('tuple', lambda *x: tuple(x))
    omegaconf.OmegaConf.register_new_resolver('range', lambda *x: range(*x))