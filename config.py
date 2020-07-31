import collections
from dataclasses import dataclass, field
from pathlib import Path


# Dataset config is adapted from
# https://github.com/google-research/google-research/blob/master/schema_guided_dst/baseline/config.py
# Config object that contains the following info:
# file_ranges: the file ranges of train, dev, and test set.
# max_num_cat_slot: Maximum allowed number of categorical trackable slots for a
# service.
# max_num_noncat_slot: Maximum allowed number of non-categorical trackable
# slots for a service.
# max_num_value_per_cat_slot: Maximum allowed number of values per categorical
# trackable slot.
# max_num_intent: Maximum allowed number of intents for a service.
DatasetConfig = collections.namedtuple("DatasetConfig", [
    "file_ranges", "max_num_cat_slot", "max_num_noncat_slot",
    "max_num_value_per_cat_slot", "max_num_intent"
])

DATASET_CONFIG = {
    "dstc8":
        DatasetConfig(
            file_ranges={
                "train": range(1, 128),
                "dev": range(1, 21),
                "test": range(1, 35)
            },
            max_num_cat_slot=6,
            max_num_noncat_slot=12,
            max_num_value_per_cat_slot=12,
            max_num_intent=4),
    "multiwoz21":
        DatasetConfig(
            file_ranges={
                "train": range(1, 18),
                "dev": range(1, 3),
                "test": range(1, 3)
            },
            max_num_cat_slot=9,
            max_num_noncat_slot=4,
            max_num_value_per_cat_slot=47,
            max_num_intent=1)
}


@dataclass
class Config:
    seed: int = 0
    use_gpu: bool = True
    stopwords: str = 'utils/stopwords.txt'
    # Stanford CoreNLP config.
    nlp: dict = field(default_factory=dict)
    # Training config
    pretrain_epoch: int = 50
    pretrain_log_variance_threshold: float = -5.0
    num_epoch: int = 100
    batch_size: int = 200
    optimizer: int = 'Adam'
    learning_rate: int = 2e-3
    momentum: float = 0.99
    # Elmo config
    elmo: dict = field(default_factory=dict)
    # Model: dsi-gm or dsi-base
    model: int = 'dsi-gm'
    # If True, compute the gradients of GMM prior parameters for tuning. Used
    # for dsi-gm only.
    prior_grad: bool = True
    # Multiplier in initialization of decoder weight.
    init_mult: float = 1.0
    # Default variance in prior gaussian.
    variance: float = 0.995
    feature_dim: int = 256
    l1_units: int = 100
    l2_units: int = 100
    hidden_dim: int = 100
    # Used for filtering diffident slot assignment.
    threshold: float = 0.5

    def __post_init__(self):
        self.nlp['server'] = 'utils/stanford-corenlp-full-2018-10-05'
        self.nlp['props'] = {}
        self.nlp['props']['annotators'] = 'tokenize, pos, ner, dcoref'
        self.nlp['props']['pipelineLanguage'] = 'en'
        self.nlp['props']['outputFormat'] = 'json'
        self.nlp['props']['parse.maxlen'] = '1000'
        self.nlp['props']['timeout'] = '500000'
        self.elmo['options_file'] = 'utils/elmo_pretrained_model/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        self.elmo['weights_file'] = 'utils/elmo_pretrained_model/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'


@dataclass
class Multiwoz21Config(Config):
    extract_sys: bool = False
    domain_num: int = 100
    slot_num: int = 300
    input_data_dir: str = "data/multiwoz21"
    vocab_path: str = 'data/multiwoz21/voc.txt'
    save_dir: str = "save/multiwoz21"
    pretrain_model_path: str = "save/multiwoz21/pretrain_model.pkl"
    model_path: str = "save/multiwoz21/model.pkl"
    answer_path: str = "save/multiwoz21/answer.pkl"
    result_path: str = "save/multiwoz21/result.json"
    prediction_dir: str = "save/multiwoz21/prediction"

    def __post_init__(self):
        super().__post_init__()
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.prediction_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class Dstc8Config(Config):
    extract_sys: bool = False
    domain_num: int = 100
    slot_num: int = 1000
    input_data_dir: str = "data/dstc8"
    vocab_path: str = 'data/dstc8/voc.txt'
    save_dir: str = "save/dstc8"
    pretrain_model_path: str = "save/dstc8/pretrain_model.pkl"
    model_path: str = "save/dstc8/model.pkl"
    answer_path: str = "save/dstc8/answer.pkl"
    result_path: str = "save/dstc8/result.json"
    prediction_dir: str = "save/dstc8/prediction"

    def __post_init__(self):
        super().__post_init__()
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.prediction_dir).mkdir(parents=True, exist_ok=True)
