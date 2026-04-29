import joblib

model_obj = joblib.load('models/model_v1.pkl')
print(type(model_obj))
print(model_obj.keys())