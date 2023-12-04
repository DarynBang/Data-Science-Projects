import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

hide_footer_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.set_page_config(layout='wide')
st.markdown(hide_footer_style, unsafe_allow_html=True)

#Import data here
data = pd.read_csv(r"https://raw.githubusercontent.com/DarynBang/Data-Science-Projects/main/Healthcare_data/healthcare_dataset.csv")
sns.set_style("dark")
df = data.copy()

def plotly_layout_legend(figure):
    figure.update_layout(
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(figure)

def plotly_layout(figure):
    figure.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=False
    )
    st.plotly_chart(figure)


df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], format='%m/%d/%Y')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], format='%m/%d/%Y')
df['Hospitalized duration (days)'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

df.drop(['Date of Admission', 'Discharge Date', 'Name', 'Doctor', 'Hospital', 'Room Number'], axis=1, inplace=True)

df['Age Groups'] = pd.cut(df['Age'], bins=[0, 22, 40, 60, 80, 100], labels=["<23", "23-40", "40-60", "60-80", ">80"])
categorical_features = df.select_dtypes(include=[object]).columns
numerical_features = df.select_dtypes(exclude=[object]).columns

fixed_df = df.copy()
fixed_df['Test Results'] = fixed_df['Test Results'].replace({'Inconclusive': 0, 'Normal': 1, 'Abnormal': 2})

from category_encoders import TargetEncoder
encoder = TargetEncoder()
fixed_df['Blood Type'] = encoder.fit_transform(fixed_df['Blood Type'].values.reshape(-1, 1), fixed_df['Test Results'])
fixed_df['Medication'] = encoder.fit_transform(fixed_df['Medication'].values.reshape(-1, 1), fixed_df['Test Results'])


##Binning age groups
fixed_df['Age Groups'] = fixed_df['Age Groups'].replace({"<23": 1, "23-40": 2, "40-60": 3, "60-80": 4, ">80": 5}).astype(int)
fixed_df.drop("Age", axis=1, inplace=True)
fixed_df['Billing Amount'] = np.log(fixed_df['Billing Amount'])

#Create one hot features from categorical features
fixed_df = pd.get_dummies(fixed_df, dtype=int, drop_first=True)

#Preprocessing data
X = fixed_df.drop('Test Results', axis=1)
y = fixed_df['Test Results']

st.session_state[['X', 'y']] = X, y


options = st.sidebar.radio("Select Info to display below", options=['Statistics', 'Visualization'])
st.title("Healthcare Data Science App")
if options == 'Statistics':
    st.subheader("Data Monitoring")
    st.write(data.head())

    st.markdown("-----")
    info = pd.DataFrame(data.isnull().sum(), columns=["Is Null"])
    info.insert(1, value=data.duplicated().sum(), column='Duplicated', allow_duplicates=True)
    info.insert(2, value=data.nunique(), column='Unique', allow_duplicates=True)
    info.insert(3, value=data.dtypes, column='Dtype', allow_duplicates=True)

    st.subheader("Data information")
    info_column_1, info_column_2 = st.columns((4, 6))
    with info_column_1:
        st.write(info)

    with info_column_2:
        st.info("Most features have an object dtype and 2 features 'Date of Admission' and 'Discharge Date' aren't in the correct format ")
    df.drop_duplicates(inplace=True)

    st.write(data.describe())

    st.markdown("------")
    st.subheader("Cleaned Data without irrelevant features")
    st.write(df.sample(10))

    st.markdown("-----")
    st.subheader("Categorical features")

    cat_c1, cat_c2 = st.columns((4, 6))
    with cat_c1:
        category = st.selectbox("Select category to monitor:", options=categorical_features)
        st.write(df[category].value_counts())

    with cat_c2:
        cat_bar = px.bar(data_frame=df[category].value_counts(ascending=True), orientation='h')
        plotly_layout(cat_bar)

    category_data = df[categorical_features].nunique().sort_values()
    cat_bar = px.bar(data_frame=category_data, text_auto=True, orientation='h', title='Category unique values')
    cat_bar.update_traces(textposition='outside', marker_color='gray')
    plotly_layout(cat_bar)

elif options == 'Visualization':
    st.subheader("Visualizations")
    st.markdown("Age group distributions")

    age_dist = px.bar(data_frame=df['Age Groups'].value_counts().reset_index(), y="count", text_auto=True, color="Age Groups",
                      x='Age Groups')
    plotly_layout(age_dist)

    st.markdown('Gender Value Count in each Age Group')
    gender_age_dist = px.bar(
        data_frame=df.groupby('Age Groups')['Gender'].value_counts().reset_index(),
        x='Age Groups',
        y='count',
        color='Gender',
        barmode='group',
        text_auto=True,
    )
    plotly_layout_legend(gender_age_dist)

    st.markdown("------")
    st.markdown("Billing Amount distribution")

    bins = st.slider("Select number of bins", max_value=60, min_value=20, value=30, step=5)
    bill_dist = px.histogram(data_frame=df, x='Billing Amount', nbins=bins)
    plotly_layout(bill_dist)

    groups = st.selectbox("Choose age group to monitor", options=df['Age Groups'].unique())

    barh_c1, barh_c2 = st.columns((4, 4))
    with barh_c1:
        labels1 = st.selectbox("Select category to monitor", options=categorical_features)
        fig1 = px.bar(data_frame=df[df['Age Groups'] == groups][labels1].value_counts().reset_index(),
                      x=labels1,
                      y='count',
                      color=labels1,
                      text_auto=True,
                      )

        plotly_layout(fig1)

    with barh_c2:
        labels2 = st.selectbox("Select another category to monitor", options=categorical_features)
        fig2 = px.bar(data_frame=df[df['Age Groups'] == groups][labels2].value_counts().reset_index(),
                      x=labels2,
                      y='count',
                      color=labels2,
                      text_auto=True,
                      )
        plotly_layout(fig2)


    st.markdown("-----")
    st.markdown("Monitoring billing amount among categories")
    labels = st.selectbox("Select category", options=categorical_features)
    bill_fig = px.bar(data_frame=df.groupby(labels)['Billing Amount'].mean().reset_index().sort_values(by='Billing Amount',
                                                                                                       ascending=False),
                      y=labels,
                      x='Billing Amount',
                      color=labels,
                      text_auto=True,
                      orientation='h')
    plotly_layout(bill_fig)

