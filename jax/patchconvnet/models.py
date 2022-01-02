from layers import PatchConvNet


def S60(attach_head=True, num_classes=1000):
    model = PatchConvNet(
        depth=60, dim=384, mlp_ratio=3, attach_head=attach_head, num_classes=num_classes
    )
    return model


def S120(attach_head=True, num_classes=1000):
    model = PatchConvNet(
        depth=120,
        dim=384,
        mlp_ratio=3,
        attach_head=attach_head,
        num_classes=num_classes,
    )
    return model


def B60(attach_head=True, num_classes=1000):
    model = PatchConvNet(
        depth=60, dim=768, mlp_ratio=4, attach_head=attach_head, num_classes=num_classes
    )
    return model


def B120(attach_head=True, num_classes=1000):
    model = PatchConvNet(
        depth=120,
        dim=768,
        mlp_ratio=4,
        attach_head=attach_head,
        num_classes=num_classes,
    )
    return model


def L60(attach_head=True, num_classes=1000):
    model = PatchConvNet(
        depth=60,
        dim=1024,
        mlp_ratio=3,
        attach_head=attach_head,
        num_classes=num_classes,
    )
    return model


def L120(attach_head=True, num_classes=1000):
    model = PatchConvNet(
        depth=120,
        dim=1024,
        mlp_ratio=3,
        attach_head=attach_head,
        num_classes=num_classes,
    )
    return model
