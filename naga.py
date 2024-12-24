import os
from datetime import datetime
from itertools import product
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

def preparation():
    date_path = datetime.now().strftime("%Y-%m-%d")
    clock_path = datetime.now().strftime("%H-%M")
    time_path = date_path + '/' + clock_path
    best_judge_list = []
    return time_path, best_judge_list

def init():
    loss_list = []
    dirs_path = 'output'
    return loss_list, dirs_path

def makedirs(dirs_path):
    os.makedirs(dirs_path, exist_ok=False)

def update_dirs_path(id, time_path, dirs_path):
    dirs_path = 'naga/' + time_path + '/' + dirs_path + str(id)
    return dirs_path

def plot_loss_history(loss_list, dirs_path):
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(dirs_path + '/loss_history.png')
    plt.close()

def loadyaml(config_path):
    params_value_list = []
    params_name_list = []
    config = OmegaConf.load(config_path)
    if 'params' in config:
        params = config.params
        for key, value in params.items():
            params_value_list.append(value)
            params_name_list.append(key)
        params_value_conbinations = list(product(*params_value_list))
    else:
        raise Exception("セクションキーに対する処理が未定義です")
    return params_value_conbinations, params_name_list

def makeyaml(dirs_path, params_name_list, params_value_list):
    params_dict = dict(zip(params_name_list, params_value_list))
    config = OmegaConf.create({})
    config.params = params_dict
    with open(dirs_path + '/params.yaml', 'w') as file:
        OmegaConf.save(config, file)

def best_study(time_path, best_judge_list):
    indexed_list = list(enumerate(best_judge_list))
    sorted_data = sorted(indexed_list, key=lambda x: x[1])
    output_file = 'naga/' + time_path + '/best_study.txt'

    with open(output_file, "w") as file:
        for _, (index, loss) in enumerate(sorted_data):
            file.write(f"フォルダ番号:{index} 誤差:{loss:.4f}\n")