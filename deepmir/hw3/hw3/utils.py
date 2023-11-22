import pickle
import shutil
import torch
import torch.nn.functional as F
import math

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def init_ckpt_dir(ckpt_dir, config_file, config_name = 'config'):
    t_path = (ckpt_dir / config_name).with_suffix(config_file.suffix)
    if not t_path.exists():
        ckpt_dir.mkdir(exist_ok=True)
        shutil.copyfile(config_file, t_path)
    else:
        # check if config is the same
        if config_file.read_text() != t_path.read_text():
            raise ValueError(f'config file {config_file} and {t_path} are not the same')

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")

def scan_checkpoint(cp_dir, prefix):
    # pattern = os.path.join(cp_dir, prefix + '????????')
    # cp_list = glob.glob(pattern)
    cp_list = list(cp_dir.glob(f'{prefix}*'))
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def load_checkpoint(filepath, device):
    assert filepath.is_file()
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def top_p(logits, thres = 0.9, temperature = 1.0):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_logits = sorted_logits / temperature
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    sorted_logits = sorted_logits * temperature
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs
