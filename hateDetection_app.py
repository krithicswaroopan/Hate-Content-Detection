import streamlit as st
from appFunc import getTweets, predictText, predictdf
import pickle

st.title("Hate Speech Detection")
url = st.text_input('Enter URL')
limit = st.slider("Select limit", min_value=1, max_value=100)
if st.button('Detect',key=1):
    # fetch tweet from URL
    username, tweets = getTweets(url, limit-1)

    # loading model
    with open('model_final.pkl', 'rb') as f:
        model = pickle.load(f)

    # Making predictions
    prediction = predictdf(model, tweets)

    for i in range(len(prediction)):
        if prediction['label'][i]==0:
                prediction['label'][i] = "No Hate Detected"
        else:
            prediction['label'][i] = "Hate Detected"

    # Display prediction
    st.table(data=prediction[['Text', 'label']])

    # Download predictions as CSV
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(prediction)

    st.download_button("Download CSV", csv, "pred.csv")

        
 # Text prediction
ip = st.text_input("Enter Text")
if st.button('Detect',key=2):
    with open('model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    res = predictText(model, [ip])
    if res==0:
        st.subheader("No Hate Detected")
    else:
        st.subheader("Hate Detected") 

