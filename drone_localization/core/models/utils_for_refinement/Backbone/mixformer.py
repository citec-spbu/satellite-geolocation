from torch import nn

from .mixformer_cvt import get_mixformer_cvt


class MixFormer(nn.Module):
    def __init__(
        self,
        vit_type="base_patch16",
        satellite_size=384,
        uav_size=128,
        pretrain_path="",
        pretrain=True,
        **kwargs,
    ):
        """Initializes the model."""
        super().__init__()
        if "cvt" in vit_type:
            self.backbone, self.embed_dim = get_mixformer_cvt(
                vit_type, pretrain_path=pretrain_path, pretrain=pretrain
            )
        else:
            raise NotImplementedError("Backbone type {} is not implemented.".format(vit_type))

    def forward(self, template, search):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, search = self.backbone(template, search)
        # search shape: (b, 384, 20, 20)
        # Forward the corner head
        return template, search
