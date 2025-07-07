import pickle
import json
import config
import numpy as np

class MedicalInsurnace():
    def __init__(self,age,bmi,children,smoker,region):
        self.age = age
        self.bmi = bmi
        self.children = children
        self.smoker= smoker
        self.region = region
    def load_model(self):
        ## Read Model 
        with open(config.model_path,"rb") as file:
            self.model = pickle.load(file)
        # read json file
        with open(config.json_path,"r") as file:
            self.json_data = json.load(file)
        # read std scale file
        with open(config.std_scale_path,"rb") as file:
            self.stdscale = pickle.load(file)

    def get_charges(self):
        self.load_model() # calling model
        test_array = np.zeros(len(self.json_data["columns"]),dtype=int)

        test_array[0] = self.children
        test_array[1] = self.json_data["smoker"][self.smoker]
        region_1= "region_" + self.region
        region_index = self.json_data["columns"].index(region_1)
        test_array[region_index] = 1

        test_array[6] = 0 if self.age <18 else 1 if self.age < 30 else 2 if self.age < 45 else 3 if self.age <60 else 4
        test_array[7]=  1 if self.age >60  else 0
        test_array[8] = 1 if self.bmi > 25 else 0
        std_array = self.stdscale.transform([test_array])
        predict_charges = self.model.predict(std_array)[0]

        return predict_charges
    

# github 
# requirements.txt
# project files :  model .pkl, utils, app.py,config.py,std_scale, labeled.json
# frontend 


