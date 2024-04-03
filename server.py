from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

class model_input(BaseModel):

     Pregnancies : int
     Glucose : int
     BloodPressure : int
     SkinThickness : int
     Insulin : int
     BMI : float
     DiabetesPedigreFunction : float
     Age : int 


# Load the saved model
diabbetes_model = pickle.load(open('Trained_Diabetes_model.sav','rb'))


#Craeting a FastAPI 

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: model_input):

     input_data = input_parameters.model_dump_json()
     input_dictionary = json.loads(input_data)

     preg = input_dictionary['Pregnancies']
     glu = input_dictionary['Glucose']
     bp = input_dictionary['BloodPressure']
     skin = input_dictionary['SkinThickness']
     insulin = input_dictionary['Insulin']
     bmi = input_dictionary['BMI']
     dpf = input_dictionary['DiabetesPedigreFunction']
     age = input_dictionary['Age']


# Making list so that we do not have to reshape the input 
     input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
     prediction = diabbetes_model.predict([input_list])

     if prediction[0] == 0:
          return 'The person is not Diabetic'
     
     else:
          return 'The person is Diabetic'


