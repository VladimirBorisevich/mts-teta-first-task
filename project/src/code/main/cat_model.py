import joblib

class CatBoostRegressor():
  def __init__(self):
      self.model = joblib.load('cat_model.pkl')

  def predict_price(self, data):
      return self.model.predict(data)
