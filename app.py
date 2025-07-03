import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# App layout
st.set_page_config(page_title="BalanceBite – Analytics Dashboard", layout="wide")
st.title("🍽️ BalanceBite – End-to-End Analytics Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Visualization", 
    "🤖 Classification", 
    "📌 Clustering", 
    "🔗 Association Rules", 
    "📈 Regression"
])


  with tab1:
    st.header("📊 Data Visualization")

    @st.cache_data
    def load_data():
        return pd.read_csv("BalanceBite_Final.csv")

    df = load_data()
    st.success("Data loaded from BalanceBite_Final.csv")

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Summary Stats")
    st.write(df.describe(include='all'))

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)



with tab2:
    st.header("🤖 Classification")
   @st.cache_data
def load_data():
    return pd.read_csv("BalanceBite_Final.csv")

df = load_data()
st.success("Data loaded from BalanceBite_Final.csv")

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
    results, roc_curves = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = classification_report(y_test, preds, output_dict=True)['weighted avg']
        from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay

classes = np.unique(y_test)
is_binary = len(classes) == 2

if hasattr(model, "predict_proba") and is_binary:
    probas = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probas)
    roc_curves[name] = (fpr, tpr)

elif hasattr(model, "predict_proba") and not is_binary:
    y_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_curves[f"{name} (class {classes[i]})"] = (fpr, tpr)

    st.dataframe(pd.DataFrame(results).T.round(3))

    selected_model = st.selectbox("Choose model for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[selected_model].predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Check for binary classification
if len(np.unique(y_test)) == 2:
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probas)
            ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve Comparison (Binary Only)")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("ROC Curve only available for binary classification.")


with tab3:
    st.header("📌 Clustering")
    uploaded_file = st.file_uploader("Upload CSV for Clustering", type=["csv"], key="cluster")
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("BalanceBite_Final.csv")

    df_numeric = df.select_dtypes(include=['int64', 'float64']).dropna(axis=1)
    X_scaled = StandardScaler().fit_transform(df_numeric)

    st.subheader("Elbow Method")
    wcss = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(2, 11), wcss, marker='o')
    ax.set_title("Elbow Curve")
    st.pyplot(fig)

    k = st.slider("Select number of clusters", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)

    st.subheader("Clustered Data")
    st.dataframe(df.head())
    st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv")

with tab4:
    st.header("🔗 Association Rule Mining")
    uploaded_file = st.file_uploader("Upload CSV with multi-item column", type=["csv"], key="assoc")
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("BalanceBite_Final.csv")

    trans_col = st.selectbox("Select column with comma-separated items", df.columns)
    transactions = df[trans_col].dropna().apply(lambda x: x.split(", ")).tolist()

    te = TransactionEncoder()
    df_encoded = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

    min_support = st.slider("Min Support", 0.01, 1.0, 0.1)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)

    frequent_items = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=min_conf)
    st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

with tab5:
    st.header("📈 Regression")
    uploaded_file = st.file_uploader("Upload CSV for Regression", type=["csv"], key="reg")
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("BalanceBite_Final.csv")

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
