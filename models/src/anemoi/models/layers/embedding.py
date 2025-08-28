import torch
import torch.nn as nn
import einops


class DynamicalVariableEmbedding(nn.Module):
    def __init__(
        self,
        data_indices: dict[str, int],
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
    ):
        """
        A PyTorch module that embeds dynamical variables using an embedding layer
        Args:
            data_indices (dict or int): A dictionary mapping variable names to their indices.
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels in the MLP.
            out_channels (int): Number of output channels for the embeddings.
        Returns:
            torch.Tensor: The output tensor after applying the embedding and MLP.

        """
        super().__init__()

        assert isinstance(data_indices, (dict, int)), "Invalid data_indices"
        assert isinstance(in_channels, int) and in_channels > 0, "Invalid in_channels"
        assert (
            isinstance(hidden_channels, int) and hidden_channels > 0
        ), "Invalid hidden_channels"
        assert (
            isinstance(out_channels, int) and out_channels > 0
        ), "Invalid out_channels"

        self.data_indices = data_indices
        self.in_channels = in_channels
        self.out_channels = out_channels

        var_id = torch.tensor(list(data_indices.values()), dtype=torch.long)
        self.register_buffer("var_id", var_id, persistent=False)

        self.embeddings = nn.Embedding(
            len(data_indices),
            out_channels,
        )
        self.mlp = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)
        self.mask_token = nn.Parameter(torch.randn(out_channels))
        self.linear = nn.Linear(out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        presence_tensor: torch.tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the DynamicalVariableEmbedding module.
        Dynamically handle missing variables, and also give the
        model the possibiltity to learn a mask token for missing variables.
        In which the model can output missing variables based on the
        context of the present variables and other domains.

        args:
            x (torch.Tensor): Input tensor of shape (B*E*G, V*T).
            presence_tensor (torch.Tensor): Presence tensor of shape (B*E*G, V*T).

        """
        multi_step = x.shape[-1] // len(self.data_indices)
        x_raw = x.view(x.shape[0], len(self.data_indices), multi_step, 1)

        embbed_variable_id = self.embeddings(self.var_id).unsqueeze(0).unsqueeze(-2)
        embbed_variable = self.norm(self.mlp(x_raw))

        _mask_token = self.mask_token.view(1, 1, 1, -1)
        _mask_token = _mask_token.to(x.device)

        embbed_variable = embbed_variable + embbed_variable_id
        out = embbed_variable * presence_tensor.unsqueeze(-1) + _mask_token * (
            1 - presence_tensor
        ).unsqueeze(-1)
        out = einops.rearrange(
            out,
            "(BGE) var time channels -> (BGE) (var time) channels",
        )

        return self.linear(out).squeeze(-1)


if __name__ == "__main__":
    ### test ###
    union_variables = 4
    presence = {
        "2t": 0,
        "10u": 1,
        "10v": 1,
        "w_500": 0,
    }

    present_tensor = torch.cat(
        [
            torch.ones(100) if pres == 1 else torch.zeros(100)
            for pres in presence.values()
        ]
    ).reshape(100, 4)
    present_tensor = torch.cat([present_tensor.unsqueeze(-1)] * 2, dim=-1)
    data_indices = {"2t": 0, "10u": 1, "10v": 2, "w_500": 3}
    in_channels = 1
    out_channels = 32

    dyncemb = DynamicalVariableEmbedding(data_indices, in_channels, out_channels)

    B, T, E, G, V = 1, 2, 1, 100, 4
    x = torch.randn(B * E * G, V * T)
    x[..., 0] = 0
    x[..., -1] = 0
    # shape = B*E*G, V*T
    # emb_id = B*E*G, V, out_channels
    out = dyncemb(x, present_tensor)
    print(out.shape)
    # print(out.shape)  # (B, T, E, G, out_channels)
