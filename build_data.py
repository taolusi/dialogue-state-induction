import argparse
from pathlib import Path

from config import Multiwoz21Config, Dstc8Config, DATASET_CONFIG
from utils.annotate import Annotator
from utils.cache_feature import FeatureExtractor
from utils.data import Vocabulary


def main(args):

    if args.task_name == "multiwoz21":
        cfg = Multiwoz21Config()
    elif args.task_name == "dstc8":
        cfg = Dstc8Config()
    else:
        raise AssertionError("Task name should be included in [multiwoz21, dstc8].")

    dataset_config = DATASET_CONFIG[args.task_name]

    # Annotate candidates from dialogs
    annotator = Annotator(cfg)
    for dataset_split in ["train", "dev", "test"]:
        dialog_paths = [Path(cfg.input_data_dir) / dataset_split /
                        f"dialogues_{i:03d}.json" \
                        for i in dataset_config.file_ranges[dataset_split]]
        print(f"Annotating {dataset_split} set....")
        annotator.process(dialog_paths)

    # Extract feature using elmo.
    extractor = FeatureExtractor(cfg)
    for dataset_split in ["train", "dev", "test"]:
        dialog_paths = [Path(cfg.input_data_dir) / dataset_split /
                        f"dialogues_{i:03d}.annotation" \
                        for i in dataset_config.file_ranges[dataset_split]]
        print(f"Extracting feature in {dataset_split} set....")
        extractor.process(dialog_paths)

    # Build a vocabulary for candidates.
    dataset = {}
    for dataset_split in ["train", "dev", "test"]:
        dialog_paths = [Path(cfg.input_data_dir) / dataset_split /
                        f"dialogues_{i:03d}.annotation" \
                        for i in dataset_config.file_ranges[dataset_split]]
        dataset[dataset_split] = dialog_paths
    vocab = Vocabulary()
    vocab.make_vocabulary(dataset, "candidates")
    vocab.save(cfg.vocab_path)
    print("Vocab length is %d" % (len(vocab.stoi)), end='\n\n')

    print("End all!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_name', type=str, choices=[
                        'multiwoz21', 'dstc8'], required=True, help="Task name: multiwoz21 or dstc8.")
    args = parser.parse_args()

    main(args)
