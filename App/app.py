import streamlit as st
import joblib
import pandas as pd
from PIL import Image
from pathlib import Path
import base64

@st.cache(allow_output_mutation=True)
def load(model_path):
    #sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return model


survived_txt = """
Our estimate will identify 70% of all surivers as such and
you are selected as a SURVIVER !
This classification has a precision of 84%
The total accuracy of this prognose is: 82%
"""
nonsurvived_txt="""
Our model identifies 90% of all people who actually died as dead, and
you are unfortunately selected as NOT TO SURVIVE !
This classification has a precision of 81%
The total accuracy of this prognose is: 82%
"""

general_congrats = """
You are selected as a SURVIVER 😊
"""
general_pity = """
You would have drowned 😥
"""
def inference(row, model, selected_model, feat_cols):
    X = pd.DataFrame([row], columns = feat_cols)
    features = pd.DataFrame(X, columns = feat_cols)
    if (model.predict(features)==1):
        return (True,survived_txt  if  (selected_model == 'model_SVC_new')  else general_congrats)

    else:
        return (False,nonsurvived_txt  if  (selected_model == 'model_SVC_new')  else general_pity)


st.markdown(
 f"""<style>
    #meme{{margin-top: 100px;}} .nonsurvived{{color: red;}} .survived{{color: green}}
        .reportview-container .main .block-container{{
        }}
        .reportview-container .main {{
            margin-top: '-40px';
        }}
</style>
""", unsafe_allow_html=True)

st.title('Titanic Survival Prediction App')
st.write('The data for the following example is originally from Kaggle and contains information about people who were on board of Titanic')
image = Image.open('data/diabetes_image.jpg')
st.image(image,caption='source: wikipedia', use_column_width=True)
st.write('Please fill in your details in the left sidebar and click on the button below to check your chances to survive titanic disaster!')


ticketclass_options = {'1': 'First Class', '2': 'Middle Class', '3':'Economy Class'}
ticketclass_options_count = {k: f"{k} ({v})" for k, v in ticketclass_options.items()}
Pclass = st.sidebar.selectbox(
        "Ticket Class",
        options=sorted(ticketclass_options.keys()),
        format_func=ticketclass_options_count.get,
)

sex_options = {'0': 'Male', '1': 'Female'}
sex_options_count = {k: f"{k} ({v})" for k, v in sex_options.items()}
Sex = st.sidebar.radio(
        "Sex",
        options=sorted(sex_options.keys()),
        format_func=sex_options_count.get,
)

Age = st.sidebar.number_input("Age in Years", 1, 150, 25, 1)
SibSp = st.sidebar.number_input("Number of Siblings/Spouse", 0, 10, 1, 1)
Parch = st.sidebar.number_input("Number of Parent/Children relationships", 0, 10, 1, 1)

ports = {'C': 'Cherbourg', 'S': 'Southhampton', 'Q': 'Queenstown'  }
ports_count = {k: f"{k} ({v})" for k, v in ports.items()}
selected_port = st.sidebar.selectbox(
        "Embarked at",
        options=sorted(ports.keys()),
        format_func=ports_count.get,
)

models = {'model_AB': 'model_AB', 'model_BC': 'model_BC', 'model_DT': 'model_DT','model_LR': 'model_LR',
 'model_NN': 'model_NN', 'model_SV': 'model_SV', 'model_SVC':'model_SVC','model_SVC_new':'model_SVC_new' }
models_count = {k: f"{k}" for k, v in models.items()}
selected_model = st.sidebar.selectbox(
        "Model",
        options=sorted(models.keys()),
        format_func=models_count.get,
)
FareC = None
Child = None
if(selected_model=='model_SVC_new'):
    FareC = st.sidebar.number_input("Fare/Person", 0, 830, 50, 10)
    Child = st.sidebar.number_input("Is Child?", 0, 1, 0, 1)

row = [int(float(Pclass)), int(float(Sex)), int(float(Age)), int(float(SibSp)), int(float(Parch)), selected_port]
if (st.button('Will you survive?')):
    feat_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

    model = load(f'models/Modelle/{selected_model}.joblib')
    if(selected_model == 'model_SVC_new'):
        feat_cols.append('FareC')
        feat_cols.append('Child')

        row.append(FareC)
        row.append(Child)
    result = inference(row, model, selected_model, feat_cols)
    if(result[0]):
        st.markdown(f'<span class="survived">{result[1]}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="nonsurvived">{result[1]}</span>', unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

header_html = "<img src='data:image/png;base64,{}' class='img-fluid' id='meme'>".format(
    img_to_bytes('data/output.jpg')
)
st.markdown(
    header_html, unsafe_allow_html=True,
)
