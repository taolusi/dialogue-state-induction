import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


def assert_not_nan(*input):
    for v in input:
        assert torch.isnan(v).sum().item() == 0


def block_enc(in_d, out_d):
    layers = [nn.Linear(in_d, out_d), nn.Softplus()]
    return layers


def block_dec(in_d, out_d):
    layers = [nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d)]
    return layers


def sampling(mean, log_var):
    epsilon = torch.randn_like(mean)
    return mean + torch.exp(log_var / 2) * epsilon


def gaussian_pdfs_log(x, means, log_vars, domain_num):
    G = []
    for c in range(domain_num):
        gpl = gaussian_pdf_log(
            x, means[c:c + 1, :], log_vars[c:c + 1, :]).view(-1, 1)
        G.append(gpl)
    return torch.cat(G, 1)


def gaussian_pdf_log(x, mean, log_var):
    '''
    Args
      x: [batch_size, feature_dim]
      mean: [batch_size, feature_dim]
      log_var: [batch_size, feature_dim]

    Returns:
      log probability of x on a gaussian distribution specified by mean and log_var
    '''
    return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var +
                             (x - mean).pow(2) / torch.exp(log_var), 1))


class Encoder(nn.Module):
    '''
    Encode for the posterior mean and log variance
    '''

    def __init__(self, input_dim: list, inter_dim: list, hid_dim: int):
        '''
        Args:
          input_dim: [vocab_len, feature_dim]
          inter_dim: [l1_units, l2_units]
          hid_dim: hidden_dim
        '''
        super(Encoder, self).__init__()

        # Encoder for one hot representation of candidates
        self.encoder_oh = nn.Sequential(
            *block_enc(
                input_dim[0], inter_dim[0]), *block_enc(
                inter_dim[0], inter_dim[1]), nn.Dropout(0.2))

        # Encoder for contextualised embedding features
        self.encoder_ce = nn.Sequential(
            *block_enc(
                input_dim[1], inter_dim[0]), *block_enc(
                inter_dim[0], inter_dim[1]), nn.Dropout(0.2))

        # From encoded to posterior mean
        self.mean = nn.Sequential(
            nn.Linear(2 * inter_dim[-1], hid_dim), nn.BatchNorm1d(hid_dim))

        # From encoded to postrior log variance
        self.log_var = nn.Sequential(
            nn.Linear(2 * inter_dim[-1], hid_dim), nn.BatchNorm1d(hid_dim))

    def forward(self, oh, ce):
        '''
        Args:
          oh: [batch_size, vocab_len]
          ce: [batch_size, seq_len, feature_dim]
        Returns:
          posterior mean: [batch_hidden_dim]
          posterior variance: [batch_size, hidden_dim]
        '''
        # [batch_size, l2_units]
        en_oh = self.encoder_oh(oh)

        # [batch_size, l2_units]
        en_ce = self.encoder_ce(ce)

        # Concatenate encoded oh and ce
        en = torch.cat([en_oh, en_ce], dim=1)  # [batch_size, l1+l2] for data

        # [batch_size, hidden_dim]
        z_mean = self.mean(en)
        z_log_var = self.log_var(en)

        return z_mean, z_log_var


class Decoder(nn.Module):
    '''
    Decode from slot to oh and ce.
    '''

    def __init__(
            self,
            slot_num: int,
            output_dim: list,
            init_mult: float):
        '''
        Args:
          slot_num: int
          output_dim: [vocab_len, feature_dim]
          init_mult: float
        '''
        super(Decoder, self).__init__()

        # Decoder for one hot representation
        self.decoder_oh = nn.Sequential(
            *block_dec(slot_num, output_dim[0]))

        # Decoder for contextualised embedding features
        self.decoder_ce_mean = nn.Sequential(
            *block_dec(slot_num, output_dim[1]))
        self.decoder_ce_log_var = nn.Sequential(
            *block_dec(slot_num, output_dim[1]))
        if init_mult != 0:
            self.decoder_oh[0].weight.data.uniform_(0, init_mult)
            self.decoder_ce_mean[0].weight.data.uniform_(0, init_mult)
            self.decoder_ce_log_var[0].weight.data.uniform_(0, init_mult)

    def forward(self, slot):
        '''
        Args:
          slot: latent vector for slot
        Returns:
          reconstructed oh and mean and log variance for features
        '''
        # do construction

        # [batch_size, vocab_len]
        recon_oh = F.softmax(
            self.decoder_oh(slot),
            dim=-1)

        # [batch_size, feature_dim]
        recon_ce_mean = self.decoder_ce_mean(
            slot)
        recon_ce_log_var = self.decoder_ce_log_var(
            slot)
        assert_not_nan(recon_oh, recon_ce_mean, recon_ce_log_var)

        return recon_oh, recon_ce_mean, recon_ce_log_var


class DSI_GM(nn.Module):
    def __init__(self, cfg):
        super(DSI_GM, self).__init__()
        self.cfg = cfg
        self.det = 1e-10

        self.encoder = Encoder([cfg.oh_dim, cfg.feature_dim], [
                               cfg.l1_units, cfg.l2_units], cfg.hidden_dim)
        self.decoder = Decoder(cfg.slot_num, [
                cfg.oh_dim, cfg.feature_dim], cfg.init_mult)

        self.z_drop = nn.Dropout(0.2)

        # Transformation from z to slot.
        self.z_slot = nn.Sequential(
            nn.Linear(
                cfg.hidden_dim,
                cfg.slot_num),
                nn.Dropout(0.2))    # [batch_size, slot_num]

        # Gradient required or not for prior pi, mean and log variance
        prior_grad = cfg.prior_grad

        # Prior pi, mean and log variance of different domains.
        self.pi = nn.Parameter(
            torch.full(
                (cfg.domain_num,
                 ),
                1.0 / cfg.domain_num),
            requires_grad=prior_grad)
        self.mean_d = nn.Parameter(
            torch.zeros(
                (cfg.domain_num,
                 cfg.hidden_dim)),
            requires_grad=prior_grad)
        self.log_var_d = nn.Parameter(
            torch.full(
                (cfg.domain_num, cfg.hidden_dim), np.log(
                    cfg.variance)), requires_grad=prior_grad)

    def pre_train(
            self,
            oh,
            ce,
            mask):
        '''
        Args:
          oh: [batch_size, oh_dim]
          ce: [batch_size, seq_len, feature_dim]
          mask: [batch_size, seq_len]
        '''
        # Features to pooled features
        x = mask.sum(dim=1)
        y = torch.ones(x.shape, device=x.device)
        mask_sum = torch.where(x == 0, y, x).unsqueeze(-1)
        # [batch_size, feature_dim]
        pooled_ce = (ce * mask.unsqueeze(-1)).sum(dim=1) / \
                    mask_sum

        # [batch_size, hidden_dim]
        z_mean, _ = self.encoder(oh, pooled_ce)

        # [batch_size, slot_num]
        slot = F.softmax(self.z_slot(z_mean), dim=-1)
        assert_not_nan(slot)

        # recon_oh:[batch_size, vocab_len]
        # recon_ce_mean, recon_ce_log_var: [batch_size, feature_dim)
        recon_oh, recon_ce_mean, recon_ce_log_var = self.decoder(
            slot)

        # NL1 for oh
        NL1 = -(oh * (recon_oh + self.det).log()
                ).sum(1)  # cross entropy loss
        # NL2 for ce
        dist = MultivariateNormal(
            loc=recon_ce_mean,
            covariance_matrix=torch.diag_embed(
                recon_ce_log_var.exp().sqrt()))
        NL2 = (-dist.log_prob(ce.transpose(0, 1)
                              ).transpose(0, 1) * mask).sum(1)

        loss = NL1 + NL2

        return loss.mean()

    def forward(self, oh, ce, mask, compute_loss=False, avg_loss=True):
        '''
        Args:
          oh: [batch_size, oh_dim]
          ce: [batch_size, seq_len, feature_dim]
          mask: [batch_size, seq_len]
        '''
        # Features to pooled features
        x = mask.sum(dim=1)
        y = torch.ones(x.shape, device=x.device)
        mask_sum = torch.where(x == 0, y, x).unsqueeze(-1)
        # [batch_size, feature_dim]
        pooled_ce = (ce * mask.unsqueeze(-1)).sum(dim=1) / \
                    mask_sum

        # [batch_size, hidden_dim]
        z_mean, z_log_var = self.encoder(oh, pooled_ce)
        assert_not_nan(z_mean, z_log_var)

        # [batch_size, hidden_dim]
        z = sampling(z_mean, z_log_var)
        z = self.z_drop(z)
        assert_not_nan(z)

        # [batch_size, slot_num]
        slot = F.softmax(self.z_slot(z_mean), dim=-1)
        assert_not_nan(slot)

        # recon_oh:[batch_size, vocab_len]
        # recon_ce_mean, recon_ce_log_var: [batch_size, feature_dim)
        recon_oh, recon_ce_mean, recon_ce_log_var = self.decoder(slot)

        # Domain vector when training, a specified domain otherwise
        if compute_loss:
            # [batch_size, domain_num]
            gamma_d = self.compute_gamma(
                self.pi, z, self.mean_d, self.log_var_d) + self.det
            gamma_d = gamma_d / (gamma_d.sum(1).view(-1, 1))
            loss = self.loss(
                oh,
                ce,
                mask,
                recon_oh,
                recon_ce_mean,
                recon_ce_log_var,
                gamma_d,
                z_mean,
                z_log_var,
                avg_loss)
            return loss
        else:
            gamma_d = self.compute_gamma(self.pi, z, self.mean_d, self.log_var_d)
            # Domain with largest probability
            domain = torch.argmax(gamma_d, dim=-1)
            slot = (slot + self.det).log()
            return domain, slot

    def loss(
            self,
            oh,
            ce,
            mask,
            recon_oh,
            recon_ce_mean,
            recon_ce_log_var,
            gamma_d,
            z_mean,
            z_log_var,
            avg=True):
        # NL1 for oh
        NL1 = -(oh * (recon_oh + self.det).log()
                ).sum(1)  # cross entropy loss
        # NL2 for ce
        dist = MultivariateNormal(
            loc=recon_ce_mean,
            covariance_matrix=torch.diag_embed(
                recon_ce_log_var.exp().sqrt()))
        NL2 = (-dist.log_prob(ce.transpose(0, 1)).transpose(0, 1) * mask).sum(1)
        NL = NL1 + NL2

        # KLD_for pi
        KLD1 = -torch.sum(gamma_d *
                          torch.log(self.pi.unsqueeze(0) / gamma_d + self.det), 1)
        # KLD2 for all domains
        logvar_division = self.log_var_d.unsqueeze(0)
        var_division = torch.exp(
            z_log_var.unsqueeze(1) -
            self.log_var_d.unsqueeze(0))
        diff = z_mean.unsqueeze(1) - self.mean_d.unsqueeze(0)
        diff_term = diff.pow(2) / torch.exp(self.log_var_d.unsqueeze(0))
        KLD21 = torch.sum(
            logvar_division + var_division + diff_term,
            2)
        KLD21 = 0.5 * torch.sum(gamma_d * KLD21, 1)
        KLD22 = -0.5 * torch.sum(1 + z_log_var, 1)
        KLD2 = KLD21 + KLD22
        KLD = KLD1 + KLD2

        loss = NL + KLD

        # in training mode, return averaged loss. In testing mode, return
        # individual loss
        if avg:
            return loss.mean()
        else:
            return loss

    def compute_gamma(self, pi, z, mean_d, log_var_d):
        '''Compute probability for each domain.'''
        # [batch_size, domain_num]
        gamma_d = torch.exp(
            torch.log(
                pi.unsqueeze(0)) +
            gaussian_pdfs_log(
                z,
                mean_d,
                log_var_d,
                self.cfg.domain_num))
        return gamma_d

    def save_cpu_model(self, path):
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()
        torch.save(state_dict, path)
        print("Saving model in %s." % path)

    def load_cpu_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        print("Loading model from %s." % path)

    def get_unnormalized_phi(self):
        return self.decoder.decoder_oh[0].weight.data.cpu().numpy().T

    def get_beta_mean(self):
        return self.decoder.decoder_ce_mean[0].weight.data.cpu().numpy().T

    def get_beta_logvar(self):
        return self.decoder.decoder_ce_log_var[0].weight.data.cpu().numpy().T
