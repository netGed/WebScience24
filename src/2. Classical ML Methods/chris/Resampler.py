from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import NearMiss, TomekLinks, RandomUnderSampler


class Resampler:

    @staticmethod
    def get_smote_resampling(x, y):
        resampler = SMOTE()
        x_smote, y_smote = resampler.fit_resample(x, y)
        return x_smote, y_smote

    @staticmethod
    def get_borderlinesmote_resampling(x, y):
        resampler = BorderlineSMOTE()
        x_bsmote, y_bsmote = resampler.fit_resample(x, y)
        return x_bsmote, y_bsmote

    @staticmethod
    def get_adasyn_resampling(x, y):
        resampler = ADASYN()
        x_ada, y_ada = resampler.fit_resample(x, y)
        return x_ada, y_ada

    @staticmethod
    def get_nearmiss_resampling(x, y):
        resampler = NearMiss(version=3, n_neighbors_ver3=3)
        x_near, y_near = resampler.fit_resample(x, y)
        return x_near, y_near

    @staticmethod
    def get_tomek_resampling(x, y):
        resampler = TomekLinks()
        x_tomek, y_tomek = resampler.fit_resample(x, y)
        return x_tomek, y_tomek

    @staticmethod
    def get_randomundersampler_resampling(x, y):
        resampler = RandomUnderSampler(sampling_strategy='auto')
        x_rus, y_rus = resampler.fit_resample(x, y)
        return x_rus, y_rus

    @staticmethod
    def get_manualhybrid_resampling(x, y):
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        undersample = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        x_smotefirst, y_smotefirst = smote.fit_resample(x, y)
        x_mhs, y_mhs = undersample.fit_resample(x_smotefirst, y_smotefirst)
        return x_mhs, y_mhs

    @staticmethod
    def get_smotetomek_resampling(x, y):
        resampler = SMOTETomek(random_state=42)
        x_smotet, y_smotet = resampler.fit_resample(x, y)
        return x_smotet, y_smotet

    @staticmethod
    def get_smoteen_resampling(x, y):
        resampler = SMOTEENN(random_state=42)
        x_smoteen, y_smoteen = resampler.fit_resample(x, y)
        return x_smoteen, y_smoteen
