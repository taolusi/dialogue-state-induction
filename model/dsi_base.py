import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


def block_enc(in_d, out_d):
    layers = [nn.Linear(in_d, out_d), nn.Softplus()]
    return layers


def sampling(mean, log_var):
    epsilon = torch.randn_like(mean)
    return mean + torch.exp(log_var / 2) * epsilon


class DSI_base(nn.Module):
    def __init__(self, cfg):
        super(DSI_base, self).__init__()
        self.cfg = cfg
        self.det = 1e-10

        # encoder for oh
        self.en_oh = nn.Sequential(
            *
            block_enc(
                cfg.oh_dim,
                cfg.l1_units),
            *
            block_enc(
                cfg.l1_units,
                cfg.l2_units))
        self.en_oh_drop = nn.Dropout(0.2)

        # encoder for ce
        self.en_ce = nn.Sequential(
            *
            block_enc(
                cfg.feature_dim,
                cfg.l1_units),
            *
            block_enc(
                cfg.l1_units,
                cfg.l2_units))
        self.en_ce_drop = nn.Dropout(0.2)

        # encoder for the variational parameters of z
        self.mean = nn.Sequential(
            nn.Linear(
                2 * cfg.l2_units,
                cfg.hidden_dim),
            nn.BatchNorm1d(
                cfg.hidden_dim))
        self.log_var = nn.Sequential(nn.Linear(2 * cfg.l2_units, cfg.hidden_dim), nn.BatchNorm1d(cfg.hidden_dim))

        self.z_drop = nn.Dropout(0.2)

        # Transformation from z to slot.
        self.z_slot = nn.Linear(cfg.hidden_dim, cfg.slot_num)  # D -> K

        # decoder for oh
        self.decoder_oh = nn.Sequential(nn.Linear(cfg.slot_num, cfg.oh_dim), nn.BatchNorm1d(cfg.oh_dim))
        # decoder for ce
        self.decoder_ce_mean = nn.Sequential(nn.Linear(cfg.slot_num, cfg.feature_dim), nn.BatchNorm1d(cfg.feature_dim))
        self.decoder_ce_log_var = nn.Sequential(nn.Linear(cfg.slot_num, cfg.feature_dim), nn.BatchNorm1d(cfg.feature_dim))
        if self.cfg.init_mult != 0:
            self.decoder_oh[0].weight.data.uniform_(0, cfg.init_mult)
            self.decoder_ce_mean[0].weight.data.uniform_(0, cfg.init_mult)
            self.decoder_ce_log_var[0].weight.data.uniform_(0, cfg.init_mult)

        # prior mean and variance
        self.prior_mean = torch.Tensor(1, cfg.hidden_dim).fill_(0)
        self.prior_var = torch.Tensor(1, cfg.hidden_dim).fill_(cfg.variance)
        if self.cfg.use_gpu:
            self.prior_mean = self.prior_mean.cuda()
            self.prior_var = self.prior_var.cuda()
        self.prior_log_var = self.prior_var.log()

    def forward(self, oh, ce, mask, compute_loss=False, avg_loss=True):
        '''
        Args:
          oh: [batch_size, vocab_len]
          ce: [batch_size, seq_len, feature_dim]
          mask: [batch_size, seq_len]
        '''
        # [batch_size, l2_units]
        en_oh = self.en_oh(oh)
        en_oh = self.en_oh_drop(en_oh)

        x = mask.sum(dim=1)
        y = torch.ones(x.shape)
        if self.cfg.use_gpu:
            y = y.cuda()
        mask_sum = torch.where(x == 0, y, x).unsqueeze(-1)
        # [batch_size, feature_dim]
        pooled_ce = (ce * mask.unsqueeze(-1)).sum(dim=1) / \
            mask_sum

        # [batch_size, l2_units]
        en_ce = self.en_ce(pooled_ce)
        en_ce = self.en_ce_drop(en_ce)
        # [batch_size, 2*l2 units]
        en = torch.cat([en_oh, en_ce], dim=1)  # [batch_size, 200] for data

        z_mean = self.mean(en)
        z_log_var = self.log_var(en)

        # [batch_size, hidden_dim]
        z = sampling(z_mean, z_log_var)
        z = self.z_drop(z)

        # [batch_size, slot_num]
        slot = F.softmax(self.z_slot(z), dim=-1)
        assert torch.isnan(slot).sum().item() == 0

        # do reconstruction
        # reconstructed dist over vocabulary
        recon_oh = F.softmax(self.decoder_oh(slot), dim=-1)
        assert torch.isnan(recon_oh).sum().item() == 0

        # reconstructed means of contextualised features
        recon_ce_mean = self.decoder_ce_mean(slot)
        assert torch.isnan(recon_ce_mean).sum().item() == 0

        # reconstructed logvariances of contextualised features
        recon_ce_log_var = self.decoder_ce_log_var(slot)
        assert torch.isnan(recon_ce_log_var).sum().item() == 0

        if compute_loss:
            loss = self.loss(oh, ce, mask, recon_oh, recon_ce_mean, recon_ce_log_var,
                                       z_mean, z_log_var, avg_loss)
            return loss
        else:
            slot = (slot + self.det).log()
            domain = torch.zeros(slot.size()[0])
            return domain, slot

    def loss(
            self,
            hcounts,
            feas,
            mask,
            recon_hcounts,
            recon_fs_mean,
            recon_fs_logvar,
            posterior_mean,
            posterior_logvar,
            avg=True):
        # NL
        NL1 = -(hcounts * (recon_hcounts + 1e-10).log()
                ).sum(1)  # cross entropy loss
        dist = MultivariateNormal(
            loc=recon_fs_mean,
            covariance_matrix=torch.diag_embed(
                recon_fs_logvar.exp().sqrt()))
        NL2 = (-dist.log_prob(feas.transpose(0, 1)).transpose(0, 1) * mask).sum(1)
        # put NL together
        NL = NL1 + NL2

        # KLD
        posterior_var = posterior_logvar.exp()
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_log_var.expand_as(posterior_mean)
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ((var_division + diff_term +
                      logvar_division).sum(1) - self.cfg.hidden_dim)
        # loss
        loss = NL + KLD
        # In training mode, return averaged loss. In testing mode, return
        # individual loss
        if avg:
            return loss.mean()
        else:
            return loss

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
        return self.decoder_oh[0].weight.data.cpu().numpy().T

    def get_beta_mean(self):
        return self.decoder_ce_mean[0].weight.data.cpu().numpy().T

    def get_beta_logvar(self):
        return self.decoder_ce_log_var[0].weight.data.cpu().numpy().T
