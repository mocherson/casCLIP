from .generalized_rcnn import GeneralizedRCNN
from .generalized_vl_rcnn import GeneralizedVLRCNN
from .generalized_vl_rcnn_cxr import GeneralizedVLRCNN_CXR, casCLIP_CXR

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "GeneralizedVLRCNN": GeneralizedVLRCNN,
                                 'GeneralizedVLRCNN_CXR': GeneralizedVLRCNN_CXR, 
                                 'casCLIP_CXR': casCLIP_CXR 
                                 }


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
