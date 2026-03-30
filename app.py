import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="MediFlow AI", page_icon="🏥", layout="wide")

# --- MODEL ENGINE (CACHED) ---
@st.cache_resource
def train_model():
    df = pd.read_csv('Symptom2Disease.csv')
    if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['text'])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df['label'])
    return tfidf, model, df['label'].unique()

tfidf, model, disease_list = train_model()

# Triage Mapping Logic
emergency = ['Bronchial Asthma', 'Hypertension', 'Dengue', 'Pneumonia']
urgent = ['Peptic Ulcer', 'Diabetes', 'Migraine', 'Jaundice', 'Malaria', 'Typhoid', 'Urinary Tract Infection']

# --- SESSION STATE INITIALIZATION ---
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {"name": "", "age": 0, "gender": "", "symptoms": "", "diagnosis": "", "priority": ""}
if 'history' not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏥 MediFlow Navigation")
page = st.sidebar.radio("Go to:", ["1. Patient Registration", "2. AI Diagnosis", "3. Appointment & PDF", "4. Medical History", "5. Help & Queries"])

# --- PAGE 1: REGISTRATION ---
if page == "1. Patient Registration":
    st.title("📋 Patient Registration")
    st.info("Step 1: Enter your basic details to start the triage process.")
    
    with st.container(border=True):
        name = st.text_input("Full Name", value=st.session_state.patient_data["name"])
        col1, col2 = st.columns(2)
        age = col1.number_input("Age", min_value=0, max_value=120, value=st.session_state.patient_data["age"])
        gender = col2.selectbox("Gender", ["Male", "Female", "Other"])
        
        if st.button("Save & Continue"):
            if name:
                st.session_state.patient_data.update({"name": name, "age": age, "gender": gender})
                st.success("Registration Saved! Move to 'AI Diagnosis' in the sidebar.")
            else:
                st.error("Please enter a name.")

# --- PAGE 2: AI DIAGNOSIS ---
elif page == "2. AI Diagnosis":
    st.title("🤖 AI Symptom Analysis")
    if not st.session_state.patient_data["name"]:
        st.warning("Please register on Page 1 first.")
    else:
        st.write(f"Patient: **{st.session_state.patient_data['name']}**")
        user_symptoms = st.text_area("Describe your symptoms in detail (e.g., 'I have a high fever and skin rash'):")
        
        if st.button("Run AI Triage"):
            vec = tfidf.transform([user_symptoms])
            pred = model.predict(vec)[0]
            
            priority = "Routine"
            if pred in emergency: priority = "Emergency"
            elif pred in urgent: priority = "Urgent"
            
            st.session_state.patient_data.update({
                "symptoms": user_symptoms,
                "diagnosis": pred,
                "priority": priority
            })
            
            st.subheader("Results")
            if priority == "Emergency": st.error(f"🚨 Priority: {priority} | Likely Condition: {pred}")
            elif priority == "Urgent": st.warning(f"⚠️ Priority: {priority} | Likely Condition: {pred}")
            else: st.success(f"✅ Priority: {priority} | Likely Condition: {pred}")

# --- PAGE 3: APPOINTMENT & PDF ---
elif page == "3. Appointment & PDF":
    st.title("📅 Appointment Confirmation")
    data = st.session_state.patient_data
    if not data["diagnosis"]:
        st.warning("No diagnosis found. Please complete Page 2.")
    else:
        st.write("### Review Details")
        st.write(f"**Name:** {data['name']} | **Diagnosis:** {data['diagnosis']}")
        
        # Simple PDF Gen
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Official Appointment Slip", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"Patient: {data['name']}", ln=True)
        pdf.cell(200, 10, f"AI Assessment: {data['diagnosis']} ({data['priority']})", ln=True)
        pdf.cell(200, 10, f"Date: {datetime.date.today()}", ln=True)
        
        if st.button("Confirm Appointment & Save History"):
            st.session_state.history.append({
                "date": str(datetime.date.today()),
                "diagnosis": data['diagnosis'],
                "symptoms": data['symptoms']
            })
            st.download_button("Download PDF Slip", data=pdf.output(dest='S').encode('latin-1'), file_name="Appointment.pdf")

# --- PAGE 4: MEDICAL HISTORY ---
elif page == "4. Medical History":
    st.title("📜 Patient Medical History")
    if not st.session_state.history:
        st.info("No previous records found for this session.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.table(history_df)

# --- PAGE 5: HELP & QUERIES ---
elif page == "5. Help & Queries":
    st.title("❓ Help & Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏥 Hospital Information")
        st.write("**OPD Timings:** 09:00 AM - 05:00 PM")
        st.write("**Emergency:** 24/7 Available")
        st.write("**Contact:** +1-800-MED-FLOW")
        st.write("**Location:** 123 Healthcare Ave, Digital City")
    
    with col2:
        st.subheader("📨 Submit a Query")
        with st.form("query"):
            email = st.text_input("Your Email")
            msg = st.text_area("How can we help?")
            if st.form_submit_button("Send"):
                st.success("Query sent to our support team!")

    st.divider()
    st.subheader("FAQ")
    st.write("**Is this a real doctor?** No, this is an AI triage tool for prioritization.")
