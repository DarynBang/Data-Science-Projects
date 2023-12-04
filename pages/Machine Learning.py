import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

hide_footer_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""

accuracies= []
recalls = []
precisions = []
model_list = []


from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
def model_evaluation(model, predictions, actual_values):
    model_list.append(model)
    accuracies.append(accuracy_score(predictions, actual_values))
    recalls.append(recall_score(predictions, actual_values, average='macro'))
    precisions.append(precision_score(predictions, actual_values, average='macro'))


def kdeplot(pred, actual_values):
    plt.clf()
    plt.figure(figsize=(4, 2))
    sns.kdeplot(pred, shade=True)
    sns.kdeplot(actual_values.values.flatten(), shade=True)
    plt.legend(['pred', 'y_test'])
    plot = plt.gcf()
    st.pyplot(plot)

st.set_page_config(layout='wide')
st.markdown(hide_footer_style, unsafe_allow_html=True)

X, y = st.session_state[['X', 'y']]

st.header("Predictive Modeling/Machine Learning")
st.markdown("Training features")
st.write(X.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def scaling_spliting_data(X, y):
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


X_train, X_test, y_train, y_test = scaling_spliting_data(X, y)


#Making predictions
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=23)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

st.markdown("-----")
st.subheader("LazyPredict Model Evaluation")
st.write(models)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

et_classifier = ExtraTreeClassifier(random_state=23)
et_classifier.fit(X_train, y_train)
et_predictions = et_classifier.predict(X_test)
model_evaluation("Extra Trees", et_predictions, y_test)


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(learning_rate=1, random_state=42)
adam_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.5, 1.0],
    'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)]
}
gridcv = GridSearchCV(estimator=ada, param_grid=adam_params, n_jobs=-1, cv=5,
                          return_train_score=True, scoring='precision')
gridcv.fit(X_train, y_train)
ada_predictions = gridcv.predict(X_test)
model_evaluation("Ada", ada_predictions, y_test)

c1, c2 = st.columns((5, 5))
with c1:
    st.markdown("----")
    st.markdown("Extra Trees")
    kdeplot(et_predictions, y_test)

with c2:
    st.markdown("----")
    st.markdown("Adaboost Classifier")
    kdeplot(ada_predictions, y_test)

results = pd.DataFrame(data={'Accuracy': accuracies, 'Precision': precisions, 'Recall': recalls}, index=model_list)
styled_results = results.style.highlight_max(axis=0, color='purple', subset=pd.IndexSlice[:, :])
st.write(styled_results)

