"""Official CAPT-UniShape model family.

The modules in this top-level package supersede the earlier hand-written
``src.models.reskan`` prototypes.  They wrap the vendored official UniShape
implementation under ``external/unishape`` and expose the two required models:

* Official-CAPT-UniShape-RBF-KANFusion
* Official-CAPT-UniShape-KANFusion-NoRBF
"""

from typing import Any

from .capt_unishape_kanfusion_no_rbf import OfficialCAPTUniShapeKANFusionNoRBF
from .capt_unishape_rbf_kanfusion import OfficialCAPTUniShapeRBFKANFusion

__all__ = [
    "OfficialCAPTUniShapeKANFusionNoRBF",
    "OfficialCAPTUniShapeRBFKANFusion",
    "build_model_from_config",
]


def build_model_from_config(config: dict[str, Any]):
    """Build the official CAPT-UniShape variant requested by a config dict."""
    use_rbf = bool(config.get("use_rbf_head", True))
    model_name = str(config.get("model_name", "")).lower()
    if model_name:
        normalized_name = model_name.replace("-", "_")
        if "no_rbf" in normalized_name:
            name_requests_rbf = False
        elif "rbf" in normalized_name:
            name_requests_rbf = True
        else:
            name_requests_rbf = use_rbf
        if bool(name_requests_rbf) != use_rbf:
            raise ValueError(
                f"Config mismatch: model_name={config.get('model_name')!r} conflicts with use_rbf_head={use_rbf}"
            )
    if use_rbf:
        return OfficialCAPTUniShapeRBFKANFusion(**config)
    return OfficialCAPTUniShapeKANFusionNoRBF(**config)
