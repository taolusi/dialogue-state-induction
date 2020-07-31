import json

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg

        options_file = self.cfg.elmo['options_file']
        weights_file = self.cfg.elmo['weights_file']
        self.encoder = Elmo(options_file, weights_file, 1, dropout=0)
        if self.cfg.use_gpu:
            self.encoder.cuda()
        self.encoder.eval()
        print(
             f"Elmo initialized with options:\n{options_file}\n{weights_file}.", end='\n\n')

    def process(self, dialog_paths):
        """ Write features to each corresponding dialog file."""
        for dialog_path in dialog_paths:
            print(f"Processing {dialog_path.stem}...")
            with dialog_path.open("r") as fin:
                dialog_annotations = json.load(fin)
                features = []
                for dialog_annotation in dialog_annotations:
                    features.extend(self.process_dialog(dialog_annotation))

                dialog_feature_path = dialog_path.with_suffix(".pt")
                torch.save({"features": features}, dialog_feature_path)

    def process_dialog(self, dialog):
        """Elmo representation is extracted for each candidate in a turn."""
        features = []
        for turn_idx, turn in enumerate(dialog["turns"]):
            tokens = [turn["tokens"]]
            token_ids = batch_to_ids(tokens)
            if self.cfg.use_gpu:
                token_ids = token_ids.cuda()
            embeddings = self.encoder(token_ids)["elmo_representations"][0].detach().cpu().data
            reps = []
            for _, candidates in turn["candidates"].items():
                for candidate in candidates:
                    rep = embeddings[0, candidate[0], :].detach()
                    reps.append(rep)
            features.append(reps)
        return features
