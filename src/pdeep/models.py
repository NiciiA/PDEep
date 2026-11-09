"""Public architecture selection; D's original inference module stays unchanged."""
from .core import require


def select_architecture(name,*,normalizers=None,**kwargs):
    require(name in {"A","B0","B1","C","D"},f"architecture {name!r} unsupported; select A, B0, B1, C or D explicitly")
    if name=="D":
        require(normalizers is None and not kwargs,"D needs no neural normalization/config")
        from .bayes import ExactGaussianD
        return ExactGaussianD()
    require(normalizers is not None,f"{name} requires saved training-distribution normalizers")
    if name in {"B0","B1"}:
        from .architecture_b import build_architecture_b
        return build_architecture_b(name,normalizers,**kwargs)
    if name=="C":
        from .architecture_c import build_architecture_c
        return build_architecture_c(normalizers,**kwargs)
    from .architecture_a import build_architecture_a
    return build_architecture_a(normalizers,**kwargs)
