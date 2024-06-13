import torch
import torch.nn as nn

from cleandiffuser.utils import Mlp


class MlpInvDynamic:
    def __init__(
            self,
            o_dim: int,
            a_dim: int,
            hidden_dim: int = 512,
            out_activation: nn.Module = nn.Identity(),
            optim_params: dict = {},
            device: str = "cpu",
    ):
        self.device = device
        self.o_dim, self.a_dim, self.hidden_dim = o_dim, a_dim, hidden_dim
        self.out_activation = out_activation
        self.optim_params = optim_params
        params = {"lr": 5e-4}
        params.update(optim_params)
        self.mlp = Mlp(
            2 * o_dim, [hidden_dim, hidden_dim], a_dim,
            nn.ReLU(), out_activation).to(device)
        self.optim = torch.optim.Adam(self.mlp.parameters(), **optim_params)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, o, o_next):
        return self.mlp(torch.cat([o, o_next], dim=-1))

    def update(self, o, a, o_next):
        self.optim.zero_grad()
        a_pred = self.forward(o, o_next)
        loss = ((a_pred - a) ** 2).mean()
        loss.backward()
        self.optim.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def predict(self, o, o_next):
        return self.forward(o, o_next)

    def __call__(self, o, o_next):
        return self.predict(o, o_next)

    def train(self):
        self.mlp.train()

    def eval(self):
        self.mlp.eval()

    def save(self, path):
        torch.save(self.mlp.state_dict(), path)

    def load(self, path):
        self.mlp.load_state_dict(torch.load(path, self.device))


class FancyMlpInvDynamic(MlpInvDynamic):
    def __init__(
            self,
            o_dim: int,
            a_dim: int,
            hidden_dim: int = 512,
            out_activation: nn.Module = nn.Identity(),
            optim_params: dict = {},
            device: str = "cpu",
    ):
        super().__init__(o_dim, a_dim, hidden_dim, out_activation, optim_params, device)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.o_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.Mish(),
            nn.Linear(self.hidden_dim, self.a_dim), self.out_activation).to(device)
        self.optim = torch.optim.Adam(self.mlp.parameters(), **self.optim_params)
        self._init_weights()

    def forward(self, o, o_next):
        return self.mlp(torch.cat([o, o_next - o], dim=-1))


class EnsembleMlpInvDynamic(MlpInvDynamic):
    def __init__(
            self,
            o_dim: int,
            a_dim: int,
            hidden_dim: int = 512,
            out_activation: nn.Module = nn.Identity(),
            optim_params: dict = {},
            n_models=5,
            mlp_type="standard",
            device: str = "cpu",
    ):
        assert mlp_type in ["standard", "fancy"]
        super().__init__(o_dim, a_dim, hidden_dim, out_activation, optim_params, device)
        self.n_models = n_models
        if mlp_type == "standard":
            self.mlp = nn.ModuleList([Mlp(
                2 * self.o_dim, [self.hidden_dim, self.hidden_dim], self.a_dim,
                nn.ReLU(), self.out_activation) for _ in range(n_models)]).to(device)
        else:
            self.mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(2 * self.o_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.Mish(),
                nn.Linear(self.hidden_dim, self.hidden_dim), self.out_activation) for _ in range(n_models)]).to(device)
        self.optim = torch.optim.Adam(self.mlp.parameters(), **self.optim_params)
        self._init_weights()

    def forward(self, o, o_next, idx=None):
        if idx is None:
            return sum([m(torch.cat([o, o_next], dim=-1)) for m in self.mlp]) / self.n_models
        else:
            return self.mlp[idx](torch.cat([o, o_next], dim=-1))

    def update_idx(self, idx, o, a, o_next):
        self.optim.zero_grad()
        a_pred = self.forward(o, o_next, idx)
        loss = ((a_pred - a) ** 2).mean()
        loss.backward()
        self.optim.step()
        return loss.item()
