import pickle
import pandas as pd
from pathlib import Path


def collect_stat(exp_dir: str):
    # Build a DataFrame out of single dicts
    acc = {}
    loss = {}
    for path in Path(exp_dir).rglob('result.pkl'):
        run_name = str(path).split('/')[-2]
        with open(path, 'rb') as f:
            data = pickle.load(f)
            acc[run_name] = {i: v['accuracy'] for i, v in data['server']['ServerAnalyzer'].items()}
            loss[run_name] = {i: v['loss'] for i, v in data['server']['ServerAnalyzer'].items()}
    return pd.DataFrame.from_dict(acc, orient='index'), pd.DataFrame.from_dict(loss, orient='index')

def extract_measures(data: dict, measures: list):
    return {m: {r: data[r][m] for r in data} for m in measures}

def subsample(data: dict, subsample_step: int):
    subsampled_data = {}
    for m, d in data.items():
        keys_sample = [n for n in range(0, max(d.keys())+1, subsample_step)]
        subsampled_data.update({m: {r: v for r, v in data[m].items() if r in keys_sample}})
    return subsampled_data

def process_experiment_runs(exp_dir: str, window: int = 100, overwrite: bool = False, 
                            target_accs: tuple = (60, 70, 80), subsample_step: int = 1):
    result_file = Path(exp_dir).joinpath('result.pkl')
    exp_name = result_file.parent.name
    if result_file.exists() and not overwrite:
        with open(result_file, 'rb') as f:
            data = pickle.load(f)['server']
            round_data = subsample(extract_measures(data['ServerAnalyzer'], ['accuracy', 'accuracy std']), subsample_step)
            if 'aggregated' in data:
                aggregated_data = data['aggregated']
                return round_data, pd.DataFrame.from_dict({exp_name: aggregated_data}, orient='index')
            else:
                return round_data, pd.DataFrame()
    else:
        acc, loss = collect_stat(exp_dir)
        acc_mean, acc_std, loss_mean = acc.mean(), acc.std(), loss.mean()
        acc_mean_last_w = acc.iloc[:, -window:].mean(axis=1)
        acc_mean_last_w_mean = acc_mean_last_w.mean()
        acc_mean_last_w_std = acc_mean_last_w.std()
        # Round to reach target acc
        target_accs_round = rounds_acc(acc_mean, target_accs)

        with open(result_file, 'wb') as f:
            round_data = {r: {'loss': loss_mean[r], 'accuracy': acc_mean[r], 'accuracy std': acc_std[r]} for r in
                          acc_mean.keys()}
            aggregated_data = {
                f'mean accuracy (%) of last {window} rounds': acc_mean_last_w_mean,
                f'accuracy std of last {window} rounds': acc_mean_last_w_std,
                **target_accs_round
            }
            new_data = {'server': {'ServerAnalyzer': round_data, 'aggregated': aggregated_data}}
            pickle.dump(new_data, f)
            return round_data, pd.DataFrame.from_dict({exp_name: aggregated_data}, orient='index')

def rounds_acc(acc_data: pd.DataFrame, target_accs: tuple) -> dict:
    return {f'First round reaching {t}% of accuracy': (acc_data > t).idxmax() if (acc_data > t).any() else '' for t in target_accs}

def conv_rate_speedup(conv_rates: pd.DataFrame):
    speedup = pd.DataFrame()
    for exp_name, data in conv_rates.iterrows():
        ref = '_'.join(['fedavg', *str(exp_name).split('_')[1:5]])
        ref_row = conv_rates.loc[ref]
        speedup_row = ref_row / data
        speedup = speedup.append(pd.Series(speedup_row, name=exp_name))
    return speedup

def format_speedup(conv_rates: pd.DataFrame, speedup: pd.DataFrame):
    return conv_rates.applymap(lambda x: f'${int(x)}$' if x > 0 else '-') + \
        speedup.applymap(lambda x: f' (${x:.2f}$x)'if x > 0 else ' (-)')

