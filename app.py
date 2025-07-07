from flask import Flask, jsonify,request
import config
from Project.utils import MedicalInsurnace
app = Flask(__name__)

## installation
# pip install pandas
# pip install Flask
# pip install scikit-learn

@app.route("/")
def get_home():
    return "Hello"

@app.route("/predict_charges",methods = ["POST","GET"])
def get_medical():
    if request.method == "POST":
        data = request.form
        age = int(data["age"])
        bmi = float(data["bmi"])
        children = int(data["children"])
        smoker = data["smoker"].lower()
        region = data["region"].lower()
       

        obj = MedicalInsurnace(age,bmi,children,smoker,region)
        charges = obj.get_charges()

        return jsonify({"Result" : f"Predicted medical insurance charges is {charges}"})
    

if __name__ == "__main__":
    app.run()