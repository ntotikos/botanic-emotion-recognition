"""
This module is a feature extraction factory for the purpose to cover different feature sets and feature extraction
methods. For this project, spectral, temporal, and statistical features are of interest.
"""
import pywt
import numpy as np
import torch

from src.data.data_segmentation import read_plant_file
from python_speech_features import mfcc

from src.utils.constants import SAMPLING_RATE

"""
1. For MFCC features: Read one entire teamwork session plant file, i.e. not the 1s slices, because of sliding window. 
2. For other features: 1s slices are needed as we compute manual features for these slices. LOAD 1s slices from disk. 

Wavelet
PCA
"""


class FeatureExtractor:
    def extract(self, wav_slice, method_type):
        raise NotImplementedError("Method needs to be implemented in subclass.")


class PassthroughFeatures(FeatureExtractor):
    """
    No manipulation.
    """
    def extract(self, wav_slice):
        return wav_slice


class SpectralFeatures(FeatureExtractor):
    def __init__(self, method_type):
        self.method_type = method_type

    def extract(self, wav_slice, method_type):
        spectral_features = None
        if method_type == "mfcc":
            spectral_features = mfcc(wav_slice, samplerate=10000, winlen=0.025, winstep=0.010, numcep=13)

        elif method_type == "dwt-1":
            wav_slice_np = wav_slice.numpy()
            (cA, cD) = pywt.dwt(wav_slice_np, "bior1.3")
            """
            Normalize DWT features after their computation (per sample). IMPORTANT: this is only an experiment. 
            The normalization step of the raw TS is skipped to see if the dwt computes better features in this case. 
            """

            mean_ca = np.mean(cA)
            std_dev_ca = np.std(cA)
            standardized_ca = (cA - mean_ca) / (std_dev_ca + 0.00000001)

            mean_cd = np.mean(cD)
            std_dev_cd = np.std(cD)
            standardized_cd = (cD - mean_cd) / (std_dev_cd + 0.00000001)

            spectral_features = np.concatenate([standardized_ca, standardized_cd], axis=0)

        elif method_type == "dwt-3":
            wav_slice_np = wav_slice.numpy()
            dwt_coeffs = pywt.wavedec(wav_slice_np, wavelet="bior1.3", level=3)

            normalized_coeffs = []

            for i in range(len(dwt_coeffs)):
                coeff_i = dwt_coeffs[i]

                mean_coeff_i = np.mean(coeff_i)
                std_dev_coeff_i = np.std(coeff_i)
                standardized_coeff_i = (coeff_i - mean_coeff_i) / (std_dev_coeff_i + 0.00000001)

                normalized_coeffs.append(standardized_coeff_i)

            spectral_features = np.concatenate(normalized_coeffs, axis=0)
        elif method_type == "cwt":
            wav_slice_np = wav_slice.numpy()
            downsampling_factor = 50
            downsampled_wav = wav_slice_np[::downsampling_factor]  # downsampling factor 20

            # Values were determines experimentally.
            freq_range = (1, 6)  #
            scales = SAMPLING_RATE / (downsampling_factor * 2 * np.arange(freq_range[0], freq_range[1]))

            cwt_coeffs, frequencies = pywt.cwt(downsampled_wav, scales, wavelet='morl')

            # Compute abs because it will be used for plot as well and values are more obvious.
            abs_coeffs = np.abs(cwt_coeffs)

            # Normalization
            mean_coeffs = np.mean(abs_coeffs)
            std_dev_coeffs = np.std(abs_coeffs)

            normalized_coeffs = (abs_coeffs - mean_coeffs) / (std_dev_coeffs + 0.00000001)
            spectral_features = np.concatenate(normalized_coeffs, axis=0)

            import matplotlib.pyplot as plt
            plt.imshow(normalized_coeffs, aspect='auto',
                       extent=[0, len(downsampled_wav), frequencies[-1], frequencies[0]])

            # plt.colorbar(label='Magnitude')
            # plt.xlabel('Time')
            # plt.ylabel('Frequency')
            # plt.title('Continuous Wavelet Transform')
            # plt.show()
        elif method_type == "scaleogramm":
            pass
        return spectral_features


class TemporalFeatures(FeatureExtractor):
    def extract(self, wav_slice):
        return 0


class StatisticalFeatures(FeatureExtractor):
    def extract(self, wav_slice):
        return -1


class FeatureFactory:
    @staticmethod
    def get_extractor(feature_type: str, method_type="mfcc") -> FeatureExtractor:
        if feature_type == "passthrough":
            return PassthroughFeatures()
        elif feature_type == "spectral":
            return SpectralFeatures(method_type=method_type)
        elif feature_type == "temporal":
            return TemporalFeatures()
        elif feature_type == "statistical":
            return StatisticalFeatures()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

