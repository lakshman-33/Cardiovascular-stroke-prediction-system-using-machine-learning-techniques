from django.shortcuts import render
from django.contrib import messages
# Create your views here.
from users.models import UserRegistrationModel, CardiovascularStrokeDataModel,HeartDataModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from django_pandas.io import read_frame
#%matplotlib inline
from sklearn.model_selection import train_test_split
import os
#print(os.listdir())

import warnings
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage


def AdminLogin(request):
    return render(request,'AdminLogin.html',{})


def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html',{})


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/ViewRegisterUsers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/ViewRegisterUsers.html',{'data':data})


def adminML(request):
    #gc.collect()
    dataset = HeartDataModel.objects.all()
    dataset = read_frame(dataset)
    #dataset.fillna
    print(dataset.head())
    print(type(dataset))
    print(dataset.shape)
    print(dataset.head(5))
    print(dataset.sample(5))
    print(dataset.describe())
    dataset.info()
    info = ["age", "1: male, 0: female",
            "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
            "resting blood pressure", " serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl",
            "resting electrocardiographic results (values 0,1,2)", " maximum heart rate achieved",
            "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest",
            "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy",
            "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

    for i in range(len(info)):
        print(dataset.columns[i] + ":\t\t\t" + info[i])
    #X = dataset.drop(['target'], axis=1).values
    #print("x",X)
    dataset["target"].describe()
    print(dataset["target"].unique())
    print(dataset.corr()["target"].abs().sort_values(ascending=False))
    y = dataset["target"]
    print("y",y)
    sns.countplot(y)

    print("Dataset Head",dataset.head(25))
    target_temp = dataset.target.value_counts()

    print("target Label Count=",target_temp)
    print("Percentage of patience without heart problems: " + str(round(target_temp[0] * 100 / 303, 2)))
    print("Percentage of patience with heart problems: " + str(round(target_temp[1] * 100 / 303, 2)))
 
    return render(request, 'admins/AdminHome.html', )
