import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

def predict_Loan_Approval(list1):
    df = pd.read_csv(r"C:\Users\Medhavi\Desktop\project\Loan_prediction\train_u6lujuX_CVtuZ9i.csv")

    df.drop(['Loan_ID'], axis=1, inplace=True)

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])


    category = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    encoder = LabelEncoder()
    for item in category:
        df[item] = encoder.fit_transform(df[item])
        
    df['LoanAmount_log'] = np.log(df['LoanAmount'])
    df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']
    df['TotalIncome_log']=np.log(df['TotalIncome'])
    
    LoanAmount_log = np.log(list1[10])
    TotalIncome = list1[8] + list1[9]
    TotalIncome_log=np.log(TotalIncome)
    
    
    list1[8] = LoanAmount_log
    list1[9] = TotalIncome_log
    list1.pop()
    
    X = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 
        'Credit_History', 'Property_Area', 'LoanAmount_log', 'TotalIncome_log']

    Y = ['Loan_Status']

    x_train, x_test, y_train, y_test = train_test_split(df[X], df[Y], test_size = 0.2)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    predicted_approval = model.predict([list1])

    return predicted_approval