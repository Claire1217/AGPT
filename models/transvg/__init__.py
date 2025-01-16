from .trans_vg_ca import TransVG_ca


def build_transvg_model(args):
    return TransVG_ca(args)
