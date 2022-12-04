#%%
import torch
from torch import nn
from methods.mae import imgs_to_patches

class MAEModel(nn.Module):
    def __init__(self, 
            d_model=192, 
            n_layers_encoder=12, 
            n_layers_decoder=4, 
            n_head_encoder=3, 
            n_head_decoder=3, 
            mlp_ratio=4,
            patch_size=4,
            img_size=(32, 32),
            **kwargs,
        ):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead = n_head_encoder,
                dim_feedforward=mlp_ratio*d_model,
                dropout=0,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=n_layers_encoder
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head_decoder,
                dim_feedforward=mlp_ratio*d_model,
                activation=nn.GELU(),
                batch_first=True,
                dropout=0
            ),
            num_layers=n_layers_decoder
        )

        n_tokens_in_image= img_size[0] * img_size[1] // patch_size**2
        self.positional_encoding_encoder = torch.nn.parameter.Parameter(
            torch.randn(size=(1, n_tokens_in_image, d_model))
        )
        self.input_embeddings_decoder = torch.nn.parameter.Parameter(
            torch.randn(size=(1, n_tokens_in_image, d_model))
        )
        self.cls_token_embedding = torch.nn.parameter.Parameter(
            torch.randn(size=(1, 1, d_model))
        )
        numel_of_patch=3 * patch_size**2
        self.patch_projector_in = nn.Linear(numel_of_patch, d_model)
        self.patch_projector_out = nn.Linear(d_model, numel_of_patch)
        self.d_model = d_model
        self.patch_size = patch_size

    def forward(self, imgs):
        patches = imgs_to_patches(imgs, patch_size=self.patch_size)
        source = self.patches_to_tokens(patches)
        source = self.add_cls_token(source)
        return self.encoder(source)[:, 0]

    def reconstruct(self, input_with_positional_encodings):
        source = self.add_cls_token(input_with_positional_encodings)
        target = self.input_embeddings_decoder.expand(len(source), -1, -1)

        memory = self.encoder(source)
        out_embeddings = self.decoder(target, memory)
        return self.patch_projector_out(out_embeddings)

    def patches_to_tokens(self, patches):
        x = self.patch_projector_in(patches)
        return x + self.positional_encoding_encoder

    def add_cls_token(self, x):
        cls_token = self.cls_token_embedding.expand(len(x), -1, -1)
        return torch.cat((cls_token, x), dim=1)

    def get_output_dim(self):
        return self.d_model
# %%
