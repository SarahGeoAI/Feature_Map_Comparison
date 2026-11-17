import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append('main-codes/Mamba-UNet-main/code')
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
import copy

def double_conv(inChannels, outChannels):
    return nn.Sequential(
        nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(outChannels),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(outChannels),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )

def up_conv(inChannels, outChannels):
    return nn.Sequential(
        nn.ConvTranspose2d(inChannels, outChannels, kernel_size=2, stride=2),
        nn.BatchNorm2d(outChannels),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )

class UNetDecoder(nn.Module):
    def __init__(self, encoderChn, decoderChn, botChn, nCls):
        super().__init__()
        self.decoder1up = up_conv(botChn, botChn)
        self.decoder1 = double_conv(encoderChn[3]+botChn, decoderChn[0])

        self.decoder2up = up_conv(decoderChn[0], decoderChn[0])
        self.decoder2 = double_conv(encoderChn[2]+decoderChn[0], decoderChn[1])

        self.decoder3up = up_conv(decoderChn[1], decoderChn[1])
        self.decoder3 = double_conv(encoderChn[1]+decoderChn[1], decoderChn[2])

        self.decoder4up = up_conv(decoderChn[2], decoderChn[2])
        self.decoder4 = double_conv(encoderChn[0]+decoderChn[2], decoderChn[3])

        self.classifier = nn.Conv2d(decoderChn[3], nCls, kernel_size=1)

    def forward(self, bottleneck, encoder_feats):
        # encoder_feats should be [enc1, enc2, enc3, enc4]
        encoder1, encoder2, encoder3, encoder4 = encoder_feats

        x = self.decoder1up(bottleneck)
        if x.shape[-2:] != encoder4.shape[-2:]:
            encoder4 = F.interpolate(encoder4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, encoder4], dim=1)
        x = self.decoder1(x)

        x = self.decoder2up(x)
        if x.shape[-2:] != encoder3.shape[-2:]:
            encoder3 = F.interpolate(encoder3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, encoder3], dim=1)
        x = self.decoder2(x)

        x = self.decoder3up(x)
        if x.shape[-2:] != encoder2.shape[-2:]:
            encoder2 = F.interpolate(encoder2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, encoder2], dim=1)
        x = self.decoder3(x)

        x = self.decoder4up(x)
        if x.shape[-2:] != encoder1.shape[-2:]:
            encoder1 = F.interpolate(encoder1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, encoder1], dim=1)
        x = self.decoder4(x)

        x = self.classifier(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

class HybridSwinUNetC(nn.Module):
    def __init__(self, config, img_size, num_classes):
        super().__init__()
        self.config = config

        # --- Swin Transformer Encoder ---
        self.encoder = SwinTransformerSys(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=num_classes,   # not used in this hybrid version
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

        # --- Channel configuration for decoder ---
        encoderChn = [config.MODEL.SWIN.EMBED_DIM * (2 ** i) for i in range(len(config.MODEL.SWIN.DEPTHS))]
        decoderChn = encoderChn[::-1]  # reverse for decoder
        botChn = encoderChn[-1]

        self.decoder = UNetDecoder(encoderChn, decoderChn, botChn, num_classes)

    def forward(self, x):
        """
        Forward pass:
        - Extract Swin encoder features
        - Reshape tokens to 2D feature maps
        - Pass through CNN-based UNet decoder
        """
        # Encoder forward (no modification to SwinTransformerSys)
        x, x_downsample = self.encoder.forward_features(x)  # returns bottleneck + list of encoder features

        # Convert token sequences [B, L, C] → [B, C, H, W]
        def tokens_to_features(tensor):
            B, L, C = tensor.shape
            H = W = int(L ** 0.5)
            return tensor.permute(0, 2, 1).contiguous().view(B, C, H, W)

        bottleneck = tokens_to_features(x)
        skips = [tokens_to_features(f) for f in x_downsample]

        # CNN-based decoder with skip connections
        out = self.decoder(bottleneck, skips)
        return out

    def load_from(self, config):
        """
        Optional: Load pretrained Swin Transformer weights
        (same logic style as HybridMambaUNet)
        """
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print(f"Loading pretrained weights from: {pretrained_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device)

            if "model" not in pretrained_dict:
                print("--- Loading pretrained model directly ---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print(f"Removing key: {k}")
                        del pretrained_dict[k]
                msg = self.encoder.load_state_dict(pretrained_dict, strict=False)
                return

            pretrained_dict = pretrained_dict["model"]
            print("--- Loading encoder weights from Swin checkpoint ---")
            model_dict = self.encoder.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            for k in list(full_dict.keys()):
                if k in model_dict and full_dict[k].shape != model_dict[k].shape:
                    print(f"Skipping: {k} (mismatch {full_dict[k].shape} vs {model_dict[k].shape})")
                    del full_dict[k]

            msg = self.encoder.load_state_dict(full_dict, strict=False)
        else:
            print("No pretrained checkpoint found.")