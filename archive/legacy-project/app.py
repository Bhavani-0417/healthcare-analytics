import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
@st.cache  # This decorator will cache the data for better performance
def load_data():
    # Replace with your actual data loading code
    df = pd.read_csv('train.csv')  # Adjust the file path as necessary
    return df

def main():
    st.title('Hospital Stay Prediction')

    # Load data
    df = load_data()

    # Display a sample of the dataset
    st.subheader('Sample of the Dataset')
    st.write(df.head())

    # Sidebar for user inputs
    st.sidebar.subheader('Choose Input Parameters:')
    case_id = st.sidebar.text_input('Case ID')
    hospital_code = st.sidebar.text_input('Hospital Code')
    severity_illness = st.sidebar.selectbox('Severity of Illness', df['Severity of Illness'].unique())
    type_admission = st.sidebar.selectbox('Type of Admission', df['Type of Admission'].unique())

    # Prediction button
    if st.sidebar.button('Predict'):
        # Check if all inputs are provided
        if case_id and hospital_code and severity_illness and type_admission:
            # Filter data based on input
            filtered_data = df[(df['case_id'] == int(case_id)) &
                               (df['Hospital_code'] == int(hospital_code)) &
                               (df['Severity of Illness'] == severity_illness) &
                               (df['Type of Admission'] == type_admission)]

            if not filtered_data.empty:
                # Display prediction
                st.subheader('Prediction:')
                st.write('Hospital Stay:', filtered_data['Stay'].iloc[0])
            else:
                st.error('No data found for the entered details.')

        else:
            st.sidebar.warning('Please enter all input parameters.')

if __name__ == '__main__':
    main()
