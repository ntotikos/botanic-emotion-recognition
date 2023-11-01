"""
This module is a feature extraction factory for the purpose to cover different feature sets and feature extraction
methods. For this project, spectral, temporal, and statistical features are of interest.
"""
from src.data.data_segmentation import read_plant_file
from python_speech_features import mfcc

"""
1. For MFCC features: Read one entire teamwork session plant file, i.e. not the 1s slices, because of sliding window. 
2. For other features: 1s slices are needed as we compute manual features for these slices. LOAD 1s slices from disk. 

Wavelet
PCA
"""


class FeatureExtractor:
    def extract(self, wav_slice):
        raise NotImplementedError("Method needs to be implemented in subclass.")


class PassthroughFeatures(FeatureExtractor):
    """
    No manipulation.
    """
    def extract(self, wav_slice):
        return wav_slice


class SpectralFeatures(FeatureExtractor):
    def extract(self, wav_slice):
        mfcc_features = mfcc(wav_slice, samplerate=10000, winlen=0.025, winstep=0.010, numcep=13)
        #print(mfcc_features)
        return mfcc_features


class TemporalFeatures(FeatureExtractor):
    def extract(self, wav_slice):
        return 0


class StatisticalFeatures(FeatureExtractor):
    def extract(self, wav_slice):
        return -1


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
            return StatisticalFeatures()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

