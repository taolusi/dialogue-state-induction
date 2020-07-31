import json
import itertools
import os
from pathlib import Path

from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch.nn import functional as F
from torch.distributions import MultivariateNormal

from utils import pred_utils
from utils import metric
from model.dsi_base import DSI_base
from model.dsi_gm import DSI_GM

MODEL_DICT = {
    'dsi-base': DSI_base,
    'dsi-gm': DSI_GM}


class Processor(object):
    def __init__(self, cfg, dataset, vocab, dialog_paths):
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.cfg = cfg
        self.vocab = vocab
        self.dataset = dataset
        self.dialog_paths = dialog_paths

        self.cfg.oh_dim = len(self.vocab.stoi)
        self.model = MODEL_DICT[self.cfg.model](self.cfg)
        if self.cfg.use_gpu:
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.cfg.learning_rate, betas=(
                self.cfg.momentum, 0.999))

    def pre_train(self):
        if not os.path.exists(self.cfg.pretrain_model_path):
            dataset = self.dataset["train"]
            dataloader = dataset.batch_delivery(self.cfg.batch_size)
            print("Pre-training start")
            print()
            # Make optimizer for parameters only used in pretraining.
            optimizer = torch.optim.Adam(
                itertools.chain(
                    self.model.encoder.parameters(),
                    self.model.z_slot.parameters(),
                    self.model.decoder.parameters()),
                self.cfg.learning_rate,
                betas=(
                    self.cfg.momentum,
                    0.999))
            for epoch in range(self.cfg.pretrain_epoch):
                loss_epoch = 0.0
                self.model.train()
                pbar = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch+1}, pre-training progress")
                for features, candidates in pbar:
                    oh, ce, mask, _, _ = dataset.add_padding(
                        features, candidates)
                    if self.cfg.use_gpu:
                        oh = oh.cuda()
                        ce = ce.cuda()
                        mask = mask.cuda()
                    loss = self.model.pre_train(
                        oh, ce, mask)
                    # optimize
                    optimizer.zero_grad()  # clear previous gradients
                    loss.backward()  # backprop
                    optimizer.step()  # update parameters
                    # report
                    loss_epoch += loss.item()  # add loss to loss_epoch
                    pbar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
                    pbar.update(1)

            self.model.encoder.log_var.load_state_dict(
                self.model.encoder.mean.state_dict())

            # Collect predicted z_mu
            Z = []
            self.model.eval()
            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"Pre-test progress")
                for features, candidates in pbar:
                    oh, ce, mask, _, _ = dataset.add_padding(
                        features, candidates)
                    if self.cfg.use_gpu:
                        oh = oh.cuda()
                        ce = ce.cuda()
                        mask = mask.cuda()

                    x = mask.sum(dim=1)
                    y = torch.ones(x.shape, device=x.device)
                    mask_sum = torch.where(x == 0, y, x).unsqueeze(-1)
                    pooled_ce = (ce * mask.unsqueeze(-1)).sum(dim=1) / \
                        mask_sum  # [batch_size, feature_dim]

                    z_mu, _ = self.model.encoder(
                        oh, pooled_ce)
                    Z.append(z_mu)
                    pbar.update(1)

            # Collected z_mu are used to fit a GaussianMixture.
            print("Fitting a GaussianMixture...")
            Z = torch.cat(Z, 0).detach().cpu().numpy()
            gmm = GaussianMixture(
                n_components=self.cfg.domain_num,
                covariance_type='diag')
            gmm.fit_predict(Z)

            # The parameters of GaussianMixture will be used as the initial parameters
            # of DSI-GM model.
            self.model.pi.data = torch.from_numpy(
                gmm.weights_).cuda().float()
            self.model.mean_d.data = torch.from_numpy(
                gmm.means_).cuda().float()
            self.model.log_var_d.data = torch.log(
                torch.from_numpy(gmm.covariances_).cuda().float())
            torch.save(
                self.model.state_dict(),
                self.cfg.pretrain_model_path)

        else:
            self.model.load_state_dict(
                torch.load(self.cfg.pretrain_model_path))
            print("Loading from {}".format(self.cfg.pretrain_model_path))
            print()

        # Domain log variance cut by a threshold.
        if self.model.log_var_d.min() < self.cfg.pretrain_log_variance_threshold:
            threshold = torch.FloatTensor(
                [self.cfg.pretrain_log_variance_threshold])
            if self.cfg.use_gpu:
                threshold = threshold.cuda()
            self.model.log_var_d.data = torch.where(
                self.model.log_var_d > threshold, self.model.log_var_d, threshold)
            print(
                f"Performing threshold cut on log variance of pretrained normal distributions, threshold: {self.cfg.pretrain_log_variance_threshold}")
            print()

        print("Pre-training done!", end='\n\n')

    def train(self):
        dataset = self.dataset["train"]
        dataloader = dataset.batch_delivery(self.cfg.batch_size)
        prev_min_loss = 1 << 30
        for epoch in range(self.cfg.num_epoch):
            loss_epoch = 0.0
            self.model.train()  # switch to training mode
            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}, training progress")
            for features, candidates in pbar:
                oh, ce, mask, lens, _ = dataset.add_padding(
                    features, candidates)
                if self.cfg.use_gpu:
                    oh = oh.cuda()
                    ce = ce.cuda()
                    mask = mask.cuda()
                loss = self.model(
                    oh, ce, mask, compute_loss=True)
                # optimize
                self.optimizer.zero_grad()  # clear previous gradients
                loss.backward(retain_graph=True)  # backprop
                self.optimizer.step()  # update parameters

                # report
                loss_epoch += loss.item()  # add loss to loss_epoch
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
                pbar.update(1)

            print(
                f'Epoch {epoch+1}, average training loss={loss_epoch/len(dataloader):.2f}',
                end=', ')

            valid_loss = self.validate()
            print(f'validation loss={valid_loss:.2f}')
            print()

            if valid_loss <= prev_min_loss:
                self.model.save_cpu_model(self.cfg.model_path)
                print()
                prev_min_loss = valid_loss
            else:
                break
            print()

        print("Training done!", end='\n\n')

    def validate(self):
        dataset = self.dataset["dev"]
        dataloader = dataset.batch_delivery(self.cfg.batch_size)
        loss_valid = 0.0
        self.model.eval()  # switch to evaluating mode

        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc=f"Validating progress")
            for features, candidates in pbar:
                oh, ce, mask, lens, _ = dataset.add_padding(
                    features, candidates)
                if self.cfg.use_gpu:
                    oh = oh.cuda()
                    ce = ce.cuda()
                    mask = mask.cuda()
                loss = self.model(
                    oh, ce, mask, compute_loss=True)
                loss_valid += loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss_valid)})
                pbar.update(1)
        return loss_valid

    def predict(self):
        dataset = self.dataset["test"]
        dataloader = dataset.batch_delivery(self.cfg.batch_size)
        self.model.eval()

        # Slot-one-hot distribution.
        # [slot_num, vocab_len]
        slot_word_dist = F.log_softmax(
            torch.FloatTensor(
                self.model.get_unnormalized_phi()),
            dim=-1)
        assert torch.isnan(slot_word_dist).sum().item() == 0

        # Slot-ce distribution.
        # [slot_num, feature_dim]
        slot_mean_dist = torch.FloatTensor(
            self.model.get_beta_mean())
        # [slot_num, emb_dim]
        slot_stdvar_dist = torch.FloatTensor(
            self.model.get_beta_logvar()).exp().sqrt()
        if self.cfg.use_gpu:
            slot_word_dist = slot_word_dist.cuda()
            slot_mean_dist = slot_mean_dist.cuda()
            slot_stdvar_dist = slot_stdvar_dist.cuda()
        slot_emb_dist = [
            MultivariateNormal(
                loc=slot_mean_dist[k],
                covariance_matrix=torch.diag_embed(
                    slot_stdvar_dist[k])) for k in range(
                self.cfg.slot_num)]

        predictions = []

        with torch.no_grad():
            pbar = tqdm(
                dataloader,
                desc=f"Validating progress")
            for features, candidates in pbar:
                oh, ce, mask, lens, candis = dataset.add_padding(
                    features, candidates)
                if self.cfg.use_gpu:
                    oh = oh.cuda()
                    ce = ce.cuda()
                    mask = mask.cuda()
                domains, slots = self.model(
                    oh, ce, mask, compute_loss=False)
                # [batch_size, padded_candi_len, slot_num]
                padded_logps = torch.stack([slots[:, k].unsqueeze(-1) +
                                            slot_emb_dist[k].log_prob(ce) +
                                            slot_word_dist[k][candis]
                                            for k in range(self.cfg.slot_num)], dim=-1).cpu()
                assert torch.isnan(padded_logps).sum().item() == 0

                # iterating over batch_size
                for i in range(oh.shape[0]):
                    domain = domains[i]
                    true_len = lens[i]
                    # [candi_len]
                    true_candis = candis[i, :true_len]
                    # [candi_len, slot_num]
                    logps = padded_logps[i, :true_len]
                    probs = torch.softmax(logps, dim=-1)

                    prediction = [{"domain": domain.item(),
                                   "slot": torch.argmax(prob).item(),
                                   "prob": prob.data.numpy(),
                                   "word": self.vocab.itos[candi_index],
                                   } for candi_index,
                                  prob in zip(true_candis, probs)]

                    predictions.append(prediction)

                pbar.update(1)
        return predictions

    def evaluate(self, turn_predictions, joint_predictions):
        turn_predictions = iter(turn_predictions)
        joint_predictions = iter(joint_predictions)
        turn_metric = metric.Metric()
        joint_metric = metric.Metric()
        total = 0
        for dialog_path in self.dialog_paths['test']:
            annotation_path = dialog_path.with_suffix(".annotation")
            with annotation_path.open("r") as f:
                dialogs = json.load(f)
                for dialog in dialogs:
                    for turn in dialog['turns']:

                        turn_prediction = next(turn_predictions)
                        states_update = pred_utils.get_states_update(
                            turn["states_update"])
                        acc = metric.compute_acc(
                            states_update, turn_prediction)
                        f1, precision, recall = metric.compute_prf(
                            states_update, turn_prediction)
                        turn_metric.update(acc, f1, precision, recall)

                        joint_prediction = next(joint_predictions)
                        states = pred_utils.get_states(turn["states"])
                        acc = metric.compute_acc(states, joint_prediction)
                        f1, precision, recall = metric.compute_prf(
                            states, joint_prediction)
                        joint_metric.update(acc, f1, precision, recall)

                        total += 1

                        turn["states_update"] = states_update
                        turn["states"] = states
                        turn["predictions"] = turn_prediction
                        turn["acc"] = metric.compute_acc(
                            states_update, turn_prediction)
                        turn.pop("tokens")
                        turn.pop("candidates")

            # Write predicted dialogs into dialog files.
            prediction_path = Path(self.cfg.prediction_dir) / \
                dialog_path.with_suffix('.json').name
            with prediction_path.open("w") as fout:
                json.dump(
                    dialogs, fout, indent=2, separators=(
                        ",", ": "), sort_keys=True)

        turn_metric.compute_scores(total)
        joint_metric.compute_scores(total)
        return turn_metric, joint_metric
