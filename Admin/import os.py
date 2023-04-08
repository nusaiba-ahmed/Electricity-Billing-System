import os
import pickle
import tensorflow
from tensorflow import keras

models_folder_path = 'F:\BRAC Undergraduate Courses\Thesis'
models = {}

for file in os.listdir(models_folder_path):
    if file.endswith(".sav"):
        file_path = os.path.join(models_folder_path, file)
        model_name = file.split(".")[0] # remove the .sav extension
        # print(model_name)
        with open(file_path, 'rb') as file:
            print(file)
            models[model_name] = pickle.load(file, fix_imports=True, encoding='ASCII')
extractedVal = [28, 28, 10, 10, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0]
# Now you can use the loaded models to predict the value of extractedVal
for model_name, model in models.items():
    prediction = model.predict(extractedVal)
print(f"Prediction using {model_name}: {prediction}")