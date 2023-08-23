"""
This module is a feature extraction factory for the purpose to cover different feature sets and feature extraction
methods. For this project, spectral, temporal, and statistical features are of interest.
"""


class FeatureExtractor:
    pass


class PassthroughFeatures(FeatureExtractor):
    """
    No operation.
    """
    pass


class SpectralFeatures(FeatureExtractor):
    pass


class TemporalFeatures(FeatureExtractor):
    pass


class StatisticalFeatures(FeatureExtractor):
    pass


class FeatureFactory:

    @staticmethod
    def get_extractor(feature_type: str) -> FeatureExtractor:
        if feature_type == "passthrough":
            return PassthroughFeatures()
        elif feature_type == "spectral":
            return SpectralFeatures()
        elif feature_type == "temporal":
            return TemporalFeatures()
        elif feature_type == "statistical":
            return StatisticalFeatures
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

