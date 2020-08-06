import argparse
from pathlib import Path

from config import Multiwoz21Config, Dstc8Config, DATASET_CONFIG
from utils.data import DataIterator, Vocabulary
from utils.process import Processor
from utils import pred_utils


def main(args):
    if args.task_name == "multiwoz21":
        cfg = Multiwoz21Config()
    elif args.task_name == "dstc8":
        cfg = Dstc8Config()
    else:
        raise AssertionError(
            "Task name should be included in [multiwoz21, dstc8].")

    if cfg.batch_size == 1:
        raise SystemExit(
            "Exit!\nBatch size can not be set to 1 for BatchNorm1d used in pytorch!")

    cfg.model = args.model

    dataset_config = DATASET_CONFIG[args.task_name]
    dialog_paths = {}
    for dataset_split in ["train", "dev", "test"]:
        dialog_paths[dataset_split] = [Path(cfg.input_data_dir) / dataset_split /
                                       f"dialogues_{i:03d}.annotation"
                                       for i in dataset_config.file_ranges[dataset_split]]

    vocab = Vocabulary()
    vocab.load(cfg.vocab_path)
    print("Vocab length is %d" % (len(vocab.stoi)), end='\n\n')

    train_iter = DataIterator(dialog_paths["train"], vocab)
    dev_iter = DataIterator(dialog_paths["dev"], vocab)
    test_iter = DataIterator(dialog_paths["test"], vocab)
    dataset = {"train": train_iter, "dev": dev_iter, "test": test_iter}

    processor = Processor(cfg, dataset, vocab, dialog_paths)

    if args.run_mode == 'train':
        if cfg.model == 'dsi-gm':
            processor.pre_train()
        processor.train()
        predictions = processor.predict()
        turn_predictions, joint_predictions = pred_utils.get_predicted_dialogs(
            predictions, dialog_paths['test'], cfg.threshold)
        turn_metric, joint_metric = processor.evaluate(
            turn_predictions, joint_predictions)
    else:
        processor.model.load_cpu_model(cfg.model_path)
        predictions = processor.predict()
        turn_predictions, joint_predictions = pred_utils.get_predicted_dialogs(
            predictions, dialog_paths['test'], cfg.threshold)
        turn_metric, joint_metric = processor.evaluate(
            turn_predictions, joint_predictions)

    print("Turn level metrics:")
    print(f"ACC: {turn_metric.acc_score:.1%}, F1: {turn_metric.f1_score:.1%}, \
            P: {turn_metric.precision_score:.1%}, R: {turn_metric.recall_score:.1%}")
    print("Joint level metrics:")
    print(
        f"ACC: {joint_metric.acc_score:.1%}, F1: {joint_metric.f1_score:.1%}, \
            P: {joint_metric.precision_score:.1%}, R: {joint_metric.recall_score:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--task_name',
        type=str,
        choices=[
            'multiwoz21',
            'dstc8'],
        required=True,
        help="Task name: multiwoz21 or dstc8.")
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=[
            'dsi-gm',
            'dsi-base'],
        default='dsi-gm',
        help="Model name: dsi-gm or dsi-base.")
    parser.add_argument(
        '-r',
        '--run_mode',
        type=str,
        choices=[
            'train',
            'predict'],
        default='train',
        help="Task name: multiwoz21 or dstc8.")
    args = parser.parse_args()

    main(args)
