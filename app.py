import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/ATCSloan.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('/content/drive/My Drive/FDP/train (1).csv')

# Extracting independent variable:
train=train.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
X = pd.get_dummies(X)
train=pd.get_dummies(train)
# Encoding the Independent Variable

from sklearn.preprocessing import StandardScaler

#df = pd.read_csv('your file here')
ss = StandardScaler()
x_train = pd.DataFrame(ss.fit_transform(x_train),columns = x_train.columns)
x_cv = pd.DataFrame(ss.fit_transform(x_cv),columns = x_cv.columns)

def predict_note_authentication(Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
  output= model.predict(sc.transform([[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]]))
  print("LOAN_STATUS", output)
  if output==[1]:
    prediction="Loan will be given"
  else:
    prediction="Loan will not be given"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Advanced Technology Consulting Service Inc</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Transform Team</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;">Loan Approval Prediction</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Loan Approval Prediction")
    
    Loan_ID= st.text_input("Loan_ID","")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Married = st.number_input('Insert Married Yes:1 No:0')
    Dependents = st.number_input('Insert Dependents 1:1 0:0 2:2 3+:3')
    Education = st.number_input('Insert Education Graduate:1 Not Graduate:0')
    Self_Employed = st.number_input('Insert Self_Employed Yes:1 No:0')
    ApplicantIncome = st.number_input("Insert ApplicantIncome",2500,7583)
    CoapplicantIncome	 = st.number_input("InsertCoapplicantIncome",0.0,2500.0)
    LoanAmount = st.number_input('Insert LoanAmount',50.0,500.0)
    Loan_Amount_Term = st.number_input('Insert Loan_Amount_Term', 10.0,500.0)
    Credit_History = st.number_input('Insert Credit_History',1.0,0.0)
    Property_Area = st.number_input('Insert Property_Area Urban:0 Rural:1 Semiurban:2')

    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Transfrom Team")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()