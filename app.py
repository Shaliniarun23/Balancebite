import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="Consumer Insights Dashboard", layout="wide")
st.title("📊 Consumer Insights Dashboard")

# Main Tab Layout
tabs = st.tabs(["📊 Data Visualization",
    "🤖 Classification",
    "📌 Clustering",
    "🔗 Association Rules",
    "📈 Regression"])

with tabs[0]:
    st.header("📊 Data Visualization")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"], key="viz")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))

        # Example: Spend by Gender
        if 'Gender' in df.columns and 'Avg Spend per Visit' in df.columns:
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='Gender', y='Avg Spend per Visit', ax=ax)
            st.subheader("Average Spend by Gender")
            st.pyplot(fig)

        # Spend Distribution
        if 'Avg Spend per Visit' in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df['Avg Spend per Visit'], bins=30, kde=True, ax=ax)
            st.subheader("Spend Distribution")
            st.pyplot(fig)


with tabs[1]:
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


with tabs[2]:
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
        ax.set_title("Elbow Curve for Optimal Clusters")
        st.pyplot(fig)

        k = st.slider("Select number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = model.fit_predict(X_scaled)
        st.subheader("Clustered Data")
        st.dataframe(df.head())
        st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv")


with tabs[3]:
    st.header("🔗 Association Rule Mining")
    uploaded_file = st.file_uploader("Upload Transactional CSV", type=["csv"], key="assoc")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        trans_col = st.selectbox("Select Transactional Column", df.columns)
        transactions = df[trans_col].dropna().apply(lambda x: x.split(", ")).tolist()

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        support = st.slider("Min Support", 0.01, 1.0, 0.1)
        confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5)

        freq_items = apriori(df_encoded, min_support=support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=confidence)
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))


with tabs[4]:
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

        st.subheader("Regression Performance Summary")
        st.dataframe(pd.DataFrame(metrics).T.round(3))
