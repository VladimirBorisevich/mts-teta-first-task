import joblib

class LightGBM():
    def __init__(self):
        self.model = joblib.load('light_model.pkl')

    def predict_price(self, data):
        return self.model.predict(data)
        