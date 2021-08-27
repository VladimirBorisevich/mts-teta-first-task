import joblib

class RobustScaler():
    def __init__(self):
        self.scaler = joblib.load('scaler.pkl')

    def get_scaled_data(self, nums):
        return self.scaler.transform(nums)
        