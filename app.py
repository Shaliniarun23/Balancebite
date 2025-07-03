import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
import io

import pandas as pd


@st.cache_data
def load_data():
    df = pd.read_csv("BalanceBite Final.csv")  # Or your full filename
    return df



df = load_data()
st.set_page_config(layout="wide")
st.title("🍽️ BalanceBite – End-to-End Analytics Dashboard")

tabs = st.tabs(["📊 Data Visualization", "🤖 Classification", "🔍 Clustering", "🔗 Association Rules", "📈 Regression"])

# -------------------- TAB 1: DATA VISUALIZATION --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Visualization", 
    "🤖 Classification", 
    "📌 Clustering", 
    "🔗 Association Rules", 
    "📈 Regression"
])
with tab1:
    st.header("📊 Data Visualization")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"], key="viz")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))

        if 'Gender' in df.columns and 'Avg Spend per Visit' in df.columns:
            st.subheader("Average Spend by Gender")
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Gender', y='Avg Spend per Visit', ax=ax)
            st.pyplot(fig)

        if 'Avg Spend per Visit' in df.columns:
            st.subheader("Spend Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['Avg Spend per Visit'], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

# -------------------- TAB 2: CLASSIFICATION --------------------
with tab2:
    st.header("🤖 Classification")
    uploaded_file = st.file_uploader("Upload CSV with target column", type=["csv"], key="clf")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Select Target Column", df.columns)
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        y = LabelEncoder().fit_transform(df[target_col])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        st.subheader("Classification Report")
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            results[name] = report['weighted avg']
        st.dataframe(pd.DataFrame(results).T)

        selected_model = st.selectbox("Choose model for Confusion Matrix", list(models.keys()))
        cm = confusion_matrix(y_test, models[selected_model].predict(X_test))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.subheader("Confusion Matrix")
        st.pyplot(fig)

# -------------------- TAB 3: CLUSTERING --------------------
with tab3:
    st.header("📌 Clustering")
    uploaded_file = st.file_uploader("Upload CSV for Clustering", type=["csv"], key="cluster")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_numeric = df.select_dtypes(include=['int64', 'float64']).dropna(axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)

        st.subheader("Elbow Method")
        wcss = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), wcss, marker='o')
        ax.set_title("Optimal Clusters (Elbow Curve)")
        st.pyplot(fig)

        k = st.slider("Select number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = model.fit_predict(X_scaled)
        st.subheader("Clustered Data")
        st.dataframe(df.head())
        st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv")

# -------------------- TAB 4: ASSOCIATION RULES --------------------
with tab4:
    st.header("🔗 Association Rule Mining")
    uploaded_file = st.file_uploader("Upload CSV with transactional column", type=["csv"], key="assoc")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        trans_col = st.selectbox("Select column with comma-separated items", df.columns)
        transactions = df[trans_col].dropna().apply(lambda x: x.split(", ")).tolist()

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        min_support = st.slider("Min Support", 0.01, 1.0, 0.1)
        min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)

        freq_items = apriori(df_encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

# -------------------- TAB 5: REGRESSION --------------------
with tab5:
    st.header("📈 Regression")
    uploaded_file = st.file_uploader("Upload CSV for Regression", type=["csv"], key="reg")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("Select Target Column", df.columns)
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Decision Tree": DecisionTreeRegressor()
        }

        metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics[name] = {
                "RMSE": mean_squared_error(y_test, preds, squared=False),
                "R2": r2_score(y_test, preds)
            }

        st.subheader("Model Performance")
        st.dataframe(pd.DataFrame(metrics).T.round(3))
