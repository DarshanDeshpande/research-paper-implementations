import flax.linen as nn
from layers import AddPositionEmbs, TransformerEncoder


class PoolFormer_S12(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(64, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(128, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(320, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(2):
            x = TransformerEncoder(512, 3, 1)(x)

        return x


class PoolFormer_S24(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(64, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(128, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(12):
            x = TransformerEncoder(320, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(4):
            x = TransformerEncoder(512, 3, 1)(x)

        return x


class PoolFormer_S36(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(64, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(64, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(128, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(128, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(320, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(18):
            x = TransformerEncoder(320, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(512, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(512, 3, 1)(x)

        return x


class PoolFormer_M36(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(96, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(96, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(192, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(192, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(384, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(18):
            x = TransformerEncoder(384, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(768, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(6):
            x = TransformerEncoder(768, 3, 1)(x)

        return x


class PoolFormer_M48(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        # H/4, W/4
        x = nn.Conv(96, (7, 7), 4, "SAME")(inputs)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(96, 3, 1)(x)

        # H/8, W/8
        x = nn.Conv(192, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(192, 3, 1)(x)

        # H/16, W/16
        x = nn.Conv(384, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(24):
            x = TransformerEncoder(384, 3, 1)(x)

        # H/32, W/32
        x = nn.Conv(768, (3, 3), 2, "SAME")(x)
        x = AddPositionEmbs()(x)
        for _ in range(8):
            x = TransformerEncoder(768, 3, 1)(x)

        return x
