
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load dataset directly
@st.cache_data
def load_data():
    return pd.read_csv("BalanceBite_Final.csv")

df = load_data()

# Setup tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Visualization",
    "🧠 Classification",
    "📌 Clustering",
    "🔗 Association Rules",
    "📈 Regression"
])

# ========================== TAB 1: DATA VISUALIZATION ==========================
with tab1:
    st.header("📊 Data Visualization")
    st.markdown("Displaying the dataset and visual patterns.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Info")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ========================== TAB 2: CLASSIFICATION ==========================
with tab2:
    st.header("🧠 Classification")
    st.markdown("Apply and compare multiple classification algorithms.")

    target_col = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col])

    if target_col and features:
        X = df[features]
        y = df[target_col]
        X = pd.get_dummies(X)
        le = LabelEncoder()
        y = le.fit_transform(y)
        y_bin = label_binarize(y, classes=np.unique(y))

        X_train, X_test, y_train, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = []
        for name, model in models.items():
            clf = OneVsRestClassifier(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            report = classification_report(y_test_bin, y_pred, output_dict=True)
            results.append([
                name,
                round(report['accuracy'], 2),
                round(report['macro avg']['precision'], 2),
                round(report['macro avg']['recall'], 2),
                round(report['macro avg']['f1-score'], 2)
            ])

        result_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
        st.dataframe(result_df)

        st.subheader("Multiclass ROC Curve")
        rf_clf = OneVsRestClassifier(RandomForestClassifier())
        rf_clf.fit(X_train, y_train)
        y_score = rf_clf.predict_proba(X_test)

        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(y_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(len(fpr)):
            ax.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Confusion Matrix")
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        y_pred_bin = rf_model.predict(X_test)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay.from_predictions(y_test_bin.argmax(axis=1), y_pred_bin.argmax(axis=1), ax=ax, cmap='Blues')
        st.pyplot(fig)

# ========================== TAB 3: CLUSTERING ==========================
with tab3:
    st.header("📌 Clustering")
    st.markdown("Coming soon...")

# ========================== TAB 4: ASSOCIATION RULES ==========================
with tab4:
    st.header("🔗 Association Rules")

    if 'Purchased_Items' in df.columns:
        transactions = df['Purchased_Items'].dropna().astype(str).apply(lambda x: x.split(','))
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_te = pd.DataFrame(te_ary, columns=te.columns_)

        frequent = apriori(df_te, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent, metric="lift", min_threshold=1.0)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        st.warning("Column 'Purchased_Items' not found in the dataset.")

# ========================== TAB 5: REGRESSION ==========================
with tab5:
    st.header("📈 Regression")
    st.markdown("Coming soon...")
