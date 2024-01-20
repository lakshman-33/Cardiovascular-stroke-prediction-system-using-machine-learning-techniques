import gc
from django.shortcuts import render
from django.contrib import messages
# Create your views here.
from users.forms import UserRegistrationForm, CardiovascularDataForm
from users.models import UserRegistrationModel, CardiovascularStrokeDataModel
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
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = UserRegistrationForm()
            return render(request, 'Register.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'Register.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})




def UserAddData(request):
    if request.method == 'POST':
        form = CardiovascularDataForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'Data Added Successfull')
            # return HttpResponseRedirect('./CustLogin')
            form = CardiovascularDataForm()
            return render(request, 'users/UserAddData.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = CardiovascularDataForm()
    return render(request, 'users/UserAddData.html', {'form': form})


def UserDataView(request):
    data_list = CardiovascularStrokeDataModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 25)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'users/DataView_list.html', {'users': users})



def CardiovascularStroke(request):
    return render(request, 'users/CardiovascularStrokeDisease.html')


def CardiovascularStrokePrediction(request):
    import pickle
    sc = pickle.load(open('sc.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    if request.method == "POST":
        chestpain = request.POST.get('chestpain')
        bloodpressure = request.POST.get('bloodpressure')
        serumcholestoral = request.POST.get('serumcholestoral')
        fastingbloodsugar = request.POST.get('fastingbloodsugar')
        electrocardiographic = request.POST.get('electrocardiographic')
        maximumheartrate = request.POST.get('maximumheartrate')
        exerciseinducedangina = request.POST.get('exerciseinducedangina')
        print(chestpain)
        print(bloodpressure)
        print(serumcholestoral)
        print(fastingbloodsugar)
        print(electrocardiographic)
        print(maximumheartrate)
        print(exerciseinducedangina)
        cp=int(chestpain)
        trestbps=int(bloodpressure)
        chol=int(serumcholestoral)
        fbs=int(fastingbloodsugar)
        restecg=int(electrocardiographic)
        thalach=int(maximumheartrate)
        exang=int(exerciseinducedangina)
        lst = []
        if cp == 0:
            lst += [1 , 0 ,0 ,0]
        elif cp == 1:
            lst += [0 ,1 ,0 ,0]
        elif cp == 2:
            lst += [0 ,0 ,1 ,0]
        elif cp >= 3:
            lst += [0 ,0 ,0 ,1]
        lst += [trestbps]
        lst += [chol]
        if fbs == 0:
            lst += [1 , 0]
        else:
            lst += [0 , 1]
        if restecg == 0:
            lst += [1 ,0 ,0]
        elif restecg == 1:
            lst += [0 ,1 ,0]
        else:
            lst += [0 , 0,1]
        lst += [thalach]
        if exang == 0:
            lst += [1 , 0]
        else:
            lst += [0 ,1 ]

        final_features = np.array([lst])
        print("final_features",final_features)
        pred = model.predict( sc.transform(final_features))


    return render(request, 'users/CardiovascularStrokePrediction.html', {"prediction":pred})


def UserMachineLearning(request):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    # warnings.simplefilter(action='ignore', category=Warning)
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")


    df.head()
    df.tail()
    df.shape
    df.isnull().sum()
    df["bmi"].isnull().sum() / df.shape[0] * 100    
    plt.bar(["Null", "Not Null"], [df["bmi"].isnull().sum(), df["bmi"].notnull().sum()])
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 3 and df[col].dtypes != "O"]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in num_cols if col not in num_but_cat]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 6 and df[col].dtypes == "O"]
    for col in cat_cols:
        print(pd.DataFrame({col: df[col].value_counts(), "Ratio": (df[col].value_counts() / len(df)) * 100}))
        sns.countplot(x=df[col], data=df)
        # plt.show(block=True)
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    for col in num_cols:
        print(df[col].describe(quantiles).T)
        df[col].hist()
        plt.xlabel(col)
        # plt.show(block=True)
    for col in num_cols:
        q1 = df[col].quantile(0.10)
        q3 = df[col].quantile(0.90)
        interquantile = q3 - q1
        up_limit = q3 + 1.5 * interquantile
        low_limit = q1 - 1.5 * interquantile
        if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
            print(col, True)
        else:
            print(col, False)
    sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, cmap="Greens")
    # plt.show()
    for col in num_cols:
        print(df.groupby("stroke").agg({col: "mean"}), end="\n\n")
    for col in cat_cols:
        print(pd.DataFrame({"target_mean": df.groupby(col)["stroke"].mean()}), end="\n\n")
    df.dropna(inplace=True)
    df["new_age_cat"] = pd.cut(x=df["age"], bins=[-1, 3, 17, 31, 46, 83], labels=["baby", "children", "young_adult", "middle_age", "old"])
    df["new_avg_glucose_level_cat"] = pd.cut(x=df["avg_glucose_level"], bins=[54, 140, 190, 272], labels=["normal", "prediabetes", "diabet"])
    df["new_bmi_cat"] = pd.cut(x=df["bmi"], bins=[10, 18.5, 24.9, 29.9, 34.9, 98], labels=["underweight", "normal", "overweight", "obese", "extremely_obese"])
    binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    cat_cols = [col for col in cat_cols if col not in "stroke"]
    ohe_cols = [col for col in df.columns if 2 < df[col].nunique() < 10]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    df.head()
    cat_cols = [col for col in df.columns if df[col].dtypes not in ["float64", "int64", "int32"]]
    num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64", "int32"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 3 and df[col].dtypes in ["float64", "int64", "int32"]]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in num_cols if col not in num_but_cat]
    cat_cols = [col for col in cat_cols if "stroke" not in col]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 6 and df[col].dtypes not in ["float64", "int64"]]
    q1 = df["bmi"].quantile(0.10)
    q3 = df["bmi"].quantile(0.90)
    interquantile = q3 - q1
    up_limit = q3 + 1.5 * interquantile
    low_limit = q1 - 1.5 * interquantile
    df.loc[(df["bmi"] < low_limit), "bmi"] = low_limit
    df.loc[(df["bmi"] > up_limit), "bmi"] = up_limit
    for col in num_cols:
        q1 = df[col].quantile(0.10)
        q3 = df[col].quantile(0.90)
        interquantile = q3 - q1
        up_limit = q3 + 1.5 * interquantile
        low_limit = q1 - 1.5 * interquantile
        if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
            print(col, True)
        else:
            print(col, False)
    rs = RobustScaler()
    df["age"] = rs.fit_transform(df[["age"]])
    df["bmi"] = rs.fit_transform(df[["bmi"]])
    df["avg_glucose_level"] = rs.fit_transform(df[["avg_glucose_level"]])
    df.head()
    
    #####################
    # Model Building
    #####################
    
    y = df["stroke"]
    X = df.drop(["stroke", "id"], axis=1)
    
    # Logistic Regression: 0.9574255041304127
    lr = LogisticRegression()
    cv = cross_val_score(lr, X, y, cv=5)
    Logistics=cv.mean()
    print(cv.mean())
    
    # KNN: 0.9547766006257383
    knn = KNeighborsClassifier()
    cv = cross_val_score(knn, X, y, cv=5)
    print(cv.mean())
    KNNs=cv.mean()
    
    # RandomForest: 0.9566106325687036
    rf = RandomForestClassifier()
    cv = cross_val_score(rf, X, y, cv=5)
    print(cv.mean())
    RandomForests=cv.mean()
    
    # SVC: 0.9574252965198238
    svc = SVC()
    cv = cross_val_score(svc, X, y, cv=5)
    print(cv.mean())
    SVMS=cv.mean()
    
    # DecisionTree: 0.9146458889989224
    dtc = DecisionTreeClassifier()
    cv = cross_val_score(dtc, X, y, cv=5)
    print(cv.mean())
    DecisionTrees=cv.mean()
    
    # GradientBoosting: 0.9549808894452851
    gb = GradientBoostingClassifier()
    cv = cross_val_score(gb, X, y, cv=5)
    print(cv.mean())    
    GradientBoostings=cv.mean()
    
    # AdaBoost: 0.9566104249581144
    ab = AdaBoostClassifier()
    cv = cross_val_score(ab, X, y, cv=5)
    print(cv.mean())
    AdaBoosts=cv.mean()
    
    # XGBoost: 0.9484623321727902
    xgb = XGBClassifier()
    cv = cross_val_score(xgb, X, y, cv=5)
    print(cv.mean())
    XGBoosts=cv.mean()
    
    # LightGBM: 0.9511099900139307
    lgbm = LGBMClassifier()
    cv = cross_val_score(lgbm, X, y, cv=5)
    print(cv.mean())
    LightGBMs=cv.mean()

    # #############################
    # # Hyperparameter Optimization
    # #############################
    
    # # RandomForest: 0.9574252965198238
    # rf_params = {"n_estimators": [100, 110, 130],
    #             "min_samples_split": [2, 5, 12, 20],
    #             "max_depth": [3, 7, 11, None],
    #             "max_features": ["auto", "sqrt", 2, 5]
    #             }
    
    # rf_best_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
    # rf_final = rf.set_params(**rf_best_grid.best_params_).fit(X, y)
    
    # cv = cross_val_score(rf_final, X, y, cv=5)
    # print(cv.mean())
    # RandomForestss=cv.mean()
    # # KNN: 0.9574252965198238
    # knn_params = {"n_neighbors": range(6, 17)}
    
    # knn_best_grid = GridSearchCV(knn, knn_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
    # knn_final = knn.set_params(**knn_best_grid.best_params_).fit(X, y)
    
    # cv = cross_val_score(knn_final, X, y, cv=5)
    # print(cv.mean())
    # KNNss=cv.mean()
    # # XGBoost: 0.9576289625076037
    
    # xgb_params = {"learning_rate": [0.1, 0.01],
    #                 "max_depth": [5, 8, 10],
    #                 "n_estimators": [100, 150, 200]}
    
    # xgb_best_grid = GridSearchCV(xgb, xgb_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
    # xgb_final = xgb.set_params(**xgb_best_grid.best_params_).fit(X, y)
    
    # cv = cross_val_score(xgb_final, X, y, cv=5)
    # print(cv.mean())
    # XGBoostss=cv.mean()
    # # LightGBM: 0.9570179645442636
    
    # lgbm_params = {"learning_rate": [0.01, 0.1],
    #                 "n_estimators": [300, 400, 500]}
    
    # lgbm_best_grid = GridSearchCV(lgbm, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
    # lgbm_final = lgbm.set_params(**lgbm_best_grid.best_params_).fit(X, y)
    
    # cv = cross_val_score(lgbm_final, X, y, cv=5)
    # print(cv.mean())
    # LightGBMss=cv.mean()    
    # #############################
    # # Result
    # #############################
    
    # # XGBoost can use with %95.76 score
    
    # ###############################
    # # Feature Importance
    # ###############################
    
    # importances = pd.DataFrame(data={"Variable": X.columns, "Importance": xgb_final.feature_importances_})
    # importances = importances.sort_values(by="Importance", ascending=False)
    # sns.barplot(x=importances["Importance"], y=importances["Variable"])
    # plt.title("XGBoost Feature Importances")
    # plt.xlabel("Variables")
    # plt.ylabel("Importance")
    # plt.show(block=True)


    scores = [Logistics, KNNs, RandomForests, SVMS, DecisionTrees, GradientBoostings,AdaBoosts,XGBoosts,LightGBMs]
    algorithms = ["LR", "KNN", "RandomForest", "SVM", "Decision Tree", "GradientBoostings","AdaBoost","XGBoost","LightGBM"]

    for i in range(len(algorithms)):
        print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + " %")
        sns.set(rc={'figure.figsize': (15, 8)})
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy score")

    sns.barplot(algorithms, scores)
    #sns.barplot(algorithms, scores, estimator = np.median, ci = 0)
    plt.show()
    dict = {
        "Logistics" :Logistics,
        "KNNs" :KNNs,
        "RandomForests" :RandomForests,
        "SVMS" :SVMS,
        "DecisionTrees" :DecisionTrees,
        "GradientBoostings" :GradientBoostings,
        "AdaBoosts" :AdaBoosts,
        "XGBoosts" :XGBoosts,
        "LightGBMs" :LightGBMs,

    }
    return render(request, 'users/Machinelearning.html', dict)
