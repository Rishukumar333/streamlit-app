
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Student Dropout Prediction in Smarter Way", page_icon="🎓", layout="wide")

# ---- Custom theme / CSS ----
st.markdown(
    """
    <style>
    :root { --bg:#0b1020; --card:#0f1724; --accent1:#9b5de5; --accent2:#00f5d4; --muted:#bfcbdc; }
    .stApp { background: linear-gradient(135deg, rgba(11,16,32,1) 0%, rgba(17,12,43,1) 50%, rgba(2,63,92,1) 100%); color:var(--muted); }
    .main-title { font-size:2.4rem; color:var(--accent2); font-weight:700; text-align:center; margin-bottom:10px; text-shadow:0 2px 10px rgba(0,0,0,0.6); }
    .metric-card { background: linear-gradient(180deg, rgba(20,26,46,0.85), rgba(15,20,36,0.6)); padding:12px; border-radius:12px; border:1px solid rgba(155,93,229,0.08); }
    .glow-btn { background: linear-gradient(90deg,var(--accent1),var(--accent2)); color:#06121b; font-weight:700; padding:8px 12px; border-radius:8px; border:none; box-shadow:0 6px 20px rgba(0,245,212,0.08); }
    .small-muted { color:#9fb0c9; font-size:0.9rem; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">🎓 Student Dropout Prediction in Smarter Way</div>', unsafe_allow_html=True)
st.write("A polished app to train multiple models, compare them, and display feature importance charts you can download for presentations.")

# Sidebar: options and upload
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel dataset", type=['csv','xlsx','xls'])

@st.cache_data
def load_df(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    return df

df = load_df(uploaded_file)
if df is None:
    st.sidebar.info("No dataset uploaded — use demo data or upload your file.")
    if st.sidebar.button("Use demo dataset"):
        np.random.seed(42)
        df = pd.DataFrame({
            'Age': np.random.randint(16,24,300),
            'CGPA': np.round(np.random.uniform(4.0,9.0,300),2),
            'Attendance_Percentage': np.random.randint(50,100,300),
            'Hours_Spent_on_LMS_per_Week': np.random.randint(0,30,300),
            'Psychological_Wellbeing_Score': np.random.randint(1,10,300),
            'Parental_Education_Level': np.random.randint(1,5,300),
            'Financial_Status': np.random.choice(['low','medium','high'],300),
            'Dropout_Risk': np.random.choice([0,1],300,p=[0.82,0.18])
        })
    else:
        st.stop()

# Basic dataset view
st.subheader("Dataset preview & info")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(8))

# Detect target candidate
possible_targets = [c for c in df.columns if 'drop' in c.lower() or 'risk' in c.lower() or 'out' in c.lower()]
default_target = possible_targets[0] if possible_targets else None
target_col = st.selectbox("Select target column (label)", options=[None]+list(df.columns), index=(0 if default_target is None else list(df.columns).index(default_target)+1))
if target_col is None:
    st.error("Please select a target column to proceed.")
    st.stop()

# Prepare features and label
y_raw = df[target_col]
if y_raw.dtype == object or y_raw.dtype.name == 'category':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))
else:
    try:
        y = y_raw.astype(int).values
        le_target = None
    except Exception:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw.astype(str))

X = df.drop(columns=[target_col]).copy()
X = X.fillna(-999)

# Identify numeric and categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

st.sidebar.subheader("Modeling Options")
algos = st.sidebar.multiselect("Choose algorithms", ['Logistic Regression','Random Forest','Gradient Boosting','Decision Tree','SVM','KNN','Naive Bayes'], default=['Logistic Regression','Random Forest'])
test_size = st.sidebar.slider("Test size (%)", 10, 50, 25)
random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

# Layout: new split — left training/results, right charts
left_col, right_col = st.columns([2,3])

with left_col:
    st.markdown("### 🔧 Train & Compare Models")
    st.markdown('<div class="metric-card">Select algorithms and press **Train selected models**</div>', unsafe_allow_html=True)
    st.write("Detected numeric columns:", numeric_cols)
    st.write("Detected categorical columns:", categorical_cols)
    
    estimators = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }
    
    # Build preprocessing
    transformers = []
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers) if transformers else None
    
    if st.button("Train selected models", key="train_btn"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=random_state, stratify=y if len(set(y))>1 else None)
        trained_models = {}
        results = {}
        feature_importances = {}
        
        for name in algos:
            clf = estimators[name]
            if preprocessor is not None:
                pipe = SKPipeline(steps=[('pre', preprocessor), ('model', clf)])
            else:
                pipe = SKPipeline(steps=[('model', clf)])
            try:
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                acc = accuracy_score(y_test, preds)
                trained_models[name] = pipe
                results[name] = {'accuracy': acc, 'preds': preds}
                
                # Try to extract feature importance or coefficients
                feat_names = []
                if preprocessor is not None:
                    # Build feature names from transformer
                    num_feats = numeric_cols if numeric_cols else []
                    cat_feats = []
                    if categorical_cols:
                        # get categories via fitted OneHotEncoder
                        enc = pipe.named_steps['pre'].named_transformers_.get('cat', None)
                        if enc is not None:
                            try:
                                cat_names = enc.get_feature_names_out(categorical_cols).tolist()
                            except Exception:
                                # fallback: use categorical column names
                                cat_names = categorical_cols
                            cat_feats = cat_names
                    feat_names = num_feats + cat_feats
                else:
                    feat_names = X.columns.tolist()
                
                imp = None
                model_obj = pipe.named_steps['model']
                if hasattr(model_obj, "feature_importances_"):
                    imp = model_obj.feature_importances_
                elif hasattr(model_obj, "coef_"):
                    # For linear models: take absolute coefficients (multi-class handled by taking mean abs)
                    coef = np.array(model_obj.coef_)
                    if coef.ndim == 1:
                        imp = np.abs(coef)
                    else:
                        imp = np.mean(np.abs(coef), axis=0)
                if imp is not None and len(imp) == len(feat_names):
                    feature_importances[name] = pd.DataFrame({'feature': feat_names, 'importance': imp}).sort_values('importance', ascending=False)
                elif imp is not None:
                    # if mismatch, attempt to align by trimming/padding
                    minlen = min(len(imp), len(feat_names))
                    feature_importances[name] = pd.DataFrame({'feature': feat_names[:minlen], 'importance': imp[:minlen]}).sort_values('importance', ascending=False)
                
            except Exception as e:
                st.warning(f"Training failed for {name}: {e}")
        
        # Save to session state
        st.session_state['trained_models'] = trained_models
        st.session_state['results'] = results
        st.session_state['feature_importances'] = feature_importances
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        
        st.success(f"✅ Trained {len(trained_models)} models. Switch to the right panel to view charts.")

    # Allow saving best model
    if 'results' in st.session_state:
        if st.button("Save best model"):
            best = max(st.session_state['results'].items(), key=lambda x: x[1]['accuracy'])[0]
            joblib.dump(st.session_state['trained_models'][best], "best_dropout_model.joblib")
            st.success(f"Saved best model to best_dropout_model.joblib ({best})")

with right_col:
    st.markdown("### 📈 Model Comparison & Charts")
    if 'results' in st.session_state and st.session_state['results']:
        res = st.session_state['results']
        # Accuracy cards
        cols = st.columns(len(res))
        for i, (name, r) in enumerate(res.items()):
            with cols[i]:
                st.markdown(f'<div class="metric-card"><h4>{name}</h4><h3 style="color:var(--accent2)">{r["accuracy"]:.3f}</h3></div>', unsafe_allow_html=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        num = len(res)
        fig, axes = plt.subplots(1, num, figsize=(5*num,4))
        if num==1:
            axes=[axes]
        for ax, (name, r) in zip(axes, res.items()):
            cm = confusion_matrix(st.session_state['y_test'], r['preds'])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
            ax.set_title(name)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        # Feature importance for each model that has it
        if 'feature_importances' in st.session_state and st.session_state['feature_importances']:
            st.subheader("🎯 Feature Importances (Top 15)")
            for name, df_imp in st.session_state['feature_importances'].items():
                st.markdown(f"**{name}**")
                top = df_imp.head(15).sort_values('importance', ascending=True)
                fig, ax = plt.subplots(figsize=(7,4))
                sns.barplot(x='importance', y='feature', data=top, ax=ax)
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                ax.set_title(f'Top features - {name}')
                plt.tight_layout()
                st.pyplot(fig)
                # Provide download button for the PNG
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                st.download_button(label=f"Download {name} feature importance PNG", data=buf, file_name=f"{name}_feature_importance.png", mime="image/png")
                plt.close(fig)
    else:
        st.info("No trained results yet. Train models from the left panel.")

# Prediction section below
st.markdown("---")
st.header("🔮 Make a prediction for a single student")
if 'trained_models' not in st.session_state:
    st.info("Train models first to enable prediction form.")
else:
    chosen_model = st.selectbox("Choose model for prediction", list(st.session_state['trained_models'].keys()))
    sample_row = X.iloc[0]
    with st.form("predict_form_v2"):
        input_vals = {}
        cols1, cols2 = st.columns(2)
        for i, col in enumerate(X.columns):
            if X[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                val = cols1.number_input(col, value=float(X[col].median()), key=f"inp_{col}")
            else:
                val = cols2.selectbox(col, options=list(X[col].astype(str).unique()), index=0, key=f"inp_{col}")
            input_vals[col] = val
        submit = st.form_submit_button("Predict")
    if submit:
        input_df = pd.DataFrame([input_vals])
        model_pipe = st.session_state['trained_models'][chosen_model]
        try:
            prob = model_pipe.predict_proba(input_df)[0][1] if hasattr(model_pipe, "predict_proba") else None
        except Exception:
            prob = None
        pred = model_pipe.predict(input_df)[0]
        st.write("Predicted class:", int(pred))
        if prob is not None:
            st.write(f"Predicted dropout probability: {prob:.2%}")
            if prob > 0.7:
                st.error("🔴 High risk - consider immediate intervention")
            elif prob > 0.3:
                st.warning("🟡 Medium risk - monitor & support")
            else:
                st.success("🟢 Low risk - continue current support")
