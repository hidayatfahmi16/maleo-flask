# Import system module
import os
import os.path
import io
import sys
# Import web framework module
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, flash, redirect, json, url_for, Response, make_response
# import data analysis module
import pandas as pd
import numpy as np
# import machine learning module
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
# Import visualization module
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
# Global variable
# Dataframe
df = pd.DataFrame()
df_null = pd.DataFrame()
df_predict = pd.DataFrame()
df_results = pd.DataFrame()
df_clustering = pd.DataFrame()
# Used in variable selection
datax = None
datay = None
xvar = None
yvar = None
cluster_var = None
# Used in variable type
num_var = []
cat_var = []
date_var = []
# Global var for machine learning
kind = None
n_clusters = 3
username = "User"
method = None
# Global for ML models
reg = None
clf = None
clu = None
labels = None
model = ""
elbow = [0]*10
# Global for reg and clf performance
y_pred = None
y_test = None
acc = None
n_data = None
matrix = None
xlabel = None
ylabel = None
# Global for reg and clf scoring
mlp_score = 0.00
gauss_score = 0.00
gboost_score = 0.00
randforest_score = 0.00
ridge_score = 0.00
bayes_score = 0.00
svm_score = 0.00
knn_score = 0.00
dec_score = 0.00
lda_score = 0.00
lr_score = 0.00
nb_score = 0.00
# Global for clu
dim = None
mark = None
# Other globals
variable = None
variables = None
uploaded_csv = False
desc = None
# Global for menu
variable_selection = False
model_selection = False

# Function for check file extensions, whether allowed or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function for create kdeplot used in histogram menu
def create_figure():
    global df,variable
    a4_dims = (6.5, 4.5)
    fig, ax = plt.subplots(figsize=a4_dims)
    if (len(cat_var) == 1):
        labels = np.unique(df[cat_var[0]].values)
        for label in labels:
            sns.kdeplot(df[df[cat_var[0]]==label][variable].values.tolist(),shade=True,ax=ax,label=label)
    else:
        y = df[variable].values.tolist()
        sns.kdeplot(y,shade=True,ax=ax)    
    return fig

# Function for create pairplot used in prediction menu
def create_figurehist():
    global df_predict,num_var
    n = len(df_predict[num_var].columns)
    g = sns.pairplot(df_predict[xvar], diag_kind="kde", height=int(8/n), aspect=1.5)  
    return g.fig

def create_regresults():
    a4_dims = (6.5, 4.5)
    fig, ax = plt.subplots(figsize=a4_dims)
    global df_results,yvar
    sns.kdeplot(df[yvar],shade=True,ax=ax)  
    return fig

def create_clfresults():
    global df_results,yvar
    a4_dims = (6.5, 4.5)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.countplot(x=yvar,data=df_results,ax=ax)
    return fig

def create_cluresults():
    global df_results
    a4_dims = (6.5, 4.5)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.countplot(x='Label',data=df_results,ax=ax)
    return fig


# Function for checking missing values number
def missingvalues(df):
    print(df)
    df_null = pd.DataFrame({'Missing Values': df.isna().sum(
    ).values, '% of Total Values': 100 * df.isnull().sum() / len(df)}, index=df.isna().sum().index)
    df_null = df_null.sort_values(
        '% of Total Values', ascending=False).round(2)
    # print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"
    # "There are " + str(df_null.shape[0]) + " columns that have missing values.")
    return df_null

# Function for imputing data using simple imputer/most frequent
def simpleimputer(df):
    imputer = SimpleImputer(strategy='most_frequent')
    if (len(cat_var) > 0):
        categories = df.select_dtypes(include=[np.object])
        category = categories.columns
        ordinal_enc_dict = {}
        # Loop over columns to encode
        for col_name in category:
            # Create ordinal encoder for the column
            ordinal_enc_dict[col_name] = OrdinalEncoder()
            # Select the nin-null values in the column
            col = df[col_name]
            col_not_null = df[col_name][col.notnull()]
            reshaped_vals = col_not_null.values.reshape(-1, 1)
            # Encode the non-null values of the column
            encoded_vals = ordinal_enc_dict[col_name].fit_transform(
                reshaped_vals)
            df.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
    if (len(date_var) != 0):
        datevar = df[date_var]
        df = df.drop(date_var, axis=1)
    df_new = pd.DataFrame((imputer.fit_transform(df)), columns=df.columns)
    for columns in df_new.columns:
        df_new[columns] = df_new[columns].astype(df[columns].dtypes)
    if (len(cat_var) > 0):
        for col_name in category:
            reshaped_col = df_new[col_name].values.reshape(-1, 1)
            df_new[col_name] = ordinal_enc_dict[col_name].inverse_transform(
                reshaped_col)
    if (len(date_var) != 0):
        df_new = df_new.join(datevar)
    return df_new

# Function for imputing data using KNN method
def knnimputer(df):
    imputer = KNNImputer()
    if (len(cat_var) > 0):
        categories = df.select_dtypes(include=[np.object])
        category = categories.columns
        ordinal_enc_dict = {}
        # Loop over columns to encode
        for col_name in category:
            # Create ordinal encoder for the column
            ordinal_enc_dict[col_name] = OrdinalEncoder()
            # Select the nin-null values in the column
            col = df[col_name]
            col_not_null = df[col_name][col.notnull()]
            reshaped_vals = col_not_null.values.reshape(-1, 1)
            # Encode the non-null values of the column
            encoded_vals = ordinal_enc_dict[col_name].fit_transform(
                reshaped_vals)
            df.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
    if (len(date_var) != 0):
        datevar = df[date_var]
        df = df.drop(date_var, axis=1)
    df_new = pd.DataFrame((imputer.fit_transform(df)), columns=df.columns)
    for columns in df_new.columns:
        df_new[columns] = df_new[columns].astype(df[columns].dtypes)
    if (len(cat_var) > 0):
        for col_name in category:
            reshaped_col = df_new[col_name].values.reshape(-1, 1)
            df_new[col_name] = ordinal_enc_dict[col_name].inverse_transform(
                reshaped_col)
    if (len(date_var) != 0):
        df_new = df_new.join(datevar)
    return df_new

# Function for imputing data using multiple regression
def miceimputer(df):
    imputer = IterativeImputer()
    if (len(cat_var) > 0):
        categories = df.select_dtypes(include=[np.object])
        category = categories.columns
        ordinal_enc_dict = {}
        # Loop over columns to encode
        for col_name in category:
            # Create ordinal encoder for the column
            ordinal_enc_dict[col_name] = OrdinalEncoder()
            # Select the nin-null values in the column
            col = df[col_name]
            col_not_null = df[col_name][col.notnull()]
            reshaped_vals = col_not_null.values.reshape(-1, 1)
            # Encode the non-null values of the column
            encoded_vals = ordinal_enc_dict[col_name].fit_transform(
                reshaped_vals)
            df.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
    if (len(date_var) != 0):
        datevar = df[date_var]
        df = df.drop(date_var, axis=1)
    df_new = pd.DataFrame((imputer.fit_transform(df)), columns=df.columns)
    for columns in df_new.columns:
        df_new[columns] = df_new[columns].astype(df[columns].dtypes)
    if(len(cat_var) > 0):
        for col_name in category:
            reshaped_col = df_new[col_name].values.reshape(-1, 1)
            df_new[col_name] = ordinal_enc_dict[col_name].inverse_transform(
                reshaped_col)
    if (len(date_var) != 0):
        df_new = df_new.join(datevar)
    return df_new

# Function for drop missing values
def dropvalue(df):
    drop_index = []
    for col in df.columns:
        ff = 100 * df.isnull().sum() / len(df)
        if ff[col] > 40:
            df = df.drop(col, axis=1)
            drop_index.append(col)
    for col in df.columns:
        ff = 100 * df.isnull().sum() / len(df)
        if ff[col] <= 40:
            df = df.dropna()
    #message = "You've dropped {} columns within above 40% of total missing values, there are {}".format(len(drop_index),drop_index)
    return(df)

# Do nothing function (hahaha)
def keeporigin(df):
    return(df)

# Regression score function, return score for every function
def MLPregr(x, y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    method = MLPRegressor()
    tuned_parameters = {'hidden_layer_sizes': [(50, 100, 50)],
                        'activation': ['relu'],
                        'alpha': [0.0001],
                        'solver': ['adam'],
                        'max_iter': [300]}

    reg = GridSearchCV(method, tuned_parameters)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

def RandomforestReg(x,y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=1)
    method = RandomForestRegressor()
    n_estimators = [100]
    max_features = ['auto']
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [True]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    reg = GridSearchCV(method, random_grid)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

def GradboostReg(x,y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=1)
    method = GradientBoostingRegressor()
    grid = {'n_estimators' : [100],
            'max_depth':[3],
            'subsample':[1.0]}
    reg = GridSearchCV(method, grid)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

def ridgeregr(x, y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    params = {'alpha': [1]}
    method = Ridge()
    reg = GridSearchCV(method, params)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

def BayesianRregr(x, y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    params = {'alpha_1': np.linspace(1e-06, 1, 6)}
    method = BayesianRidge()
    reg = GridSearchCV(method, params)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

def SVMregr(x, y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    kernels = ['rbf']
    cs = [1]
    params = {"kernel": kernels,
              "C": cs,
              'degree': [3]}
    method = SVR()
    reg = GridSearchCV(method, params)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

def KNNregr(x, y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    params = [{'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}]
    method = KNeighborsRegressor()
    reg = GridSearchCV(method, params)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

def DecisionTregr(x, y):
    global reg
    reg = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    params = {'max_features': ['auto'],
              'min_samples_split': [2],
              'min_samples_leaf': [1],
              'random_state': [123]}
    method = DecisionTreeRegressor()
    reg = GridSearchCV(method, params)
    reg.fit(X_train, y_train)
    return (reg.best_score_)

# Classification score function
def KNNclf(x, y):
    global clf
    clf = None
    method = KNeighborsClassifier()
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    params = {'n_neighbors': [3, 5],
              'leaf_size': [30],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto'],
              'n_jobs': [-1]}
    clf = GridSearchCV(method, params)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def MLPclf(x, y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    params = {
        'hidden_layer_sizes': [(100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate': ['adaptive'],
        'max_iter':[1500] }
    method = MLPClassifier()
    clf = GridSearchCV(method, params)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def DecisionTclf(x, y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    method = DecisionTreeClassifier()
    params = {'max_features': ['auto'],
              'min_samples_split': [2],
              'min_samples_leaf': [1],
              'random_state': [123]}
    clf = GridSearchCV(method, params)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def SVMclf(x, y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    method = SVC()
    params = {'C': [1],
              'kernel': ['rbf']}
    clf = GridSearchCV(method, params)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def LDAclf(x, y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    params = {  # note svd does not run with shrinkage and models using it will be tuned separately
        'store_covariance': [True, False]}
    method = LinearDiscriminantAnalysis()
    clf = GridSearchCV(method, params)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def LRclf(x, y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    solvers = ['newton-cg', 'liblinear']
    penalty = ['l2']
    c_values = [1.0]
    # define grid search
    grid = dict(solver=solvers, penalty=penalty, C=c_values, max_iter=[300])
    method = LogisticRegression()
    clf = GridSearchCV(method, grid)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def Ridgeclf(x, y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    method = RidgeClassifier()
    alpha = [1.0]
    # define grid search
    params = dict(alpha=alpha)
    clf = GridSearchCV(method, params)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def Gradboostclf(x,y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=1)
    method = GradientBoostingClassifier()
    n_estimators = [100]
    learning_rate = [0.1]
    subsample = [1.0]
    max_depth = [3]
    grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
    clf = GridSearchCV(method, grid)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

def GaussianNBclf(x, y):
    global clf
    clf = None
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    method = GaussianNB()
    params = {'var_smoothing': np.linspace(1e-9, 1, 10)}
    clf = GridSearchCV(method, params)
    clf.fit(X_train, y_train)
    return (clf.best_score_)

# Model Performance function, used if method are regression or classification
# Return ypred ytest and model accuracy 
def model_performance(x, y, method, model_name, reg, clf):
    if method == "ml1":
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, random_state=1)
        if model_name == "mlpreg":
            akurasi = round(MLPregr(X_train,y_train),4)*100
            model = "Multi-layer Perceptron"
        elif model_name == "randreg":
            akurasi = round(RandomforestReg(X_train,y_train),4)*100
            model = "Random Forest"
        elif model_name == "gradreg":
            akurasi = round(GradboostReg(X_train,y_train),4)*100
            model = "Gradient Boosting"
        elif model_name == "ridgereg":
            akurasi = round(ridgeregr(X_train,y_train),4)*100
            model = "Ridge Regression"
        elif model_name == "bayesreg":
            akurasi = round(BayesianRregr(X_train,y_train),4)*100
            model = "Bayesian Ridge Regression"
        elif model_name == "svmreg":
            akurasi = round(SVMregr(X_train,y_train),4)*100
            model = "Support Vector Machine"
        elif model_name == "knnreg":
            akurasi = round(KNNregr(X_train,y_train),4)*100
            model = "K-Nearest Neighbors"
        elif model_name == "decreg":
            akurasi = round(DecisionTregr(X_train,y_train),4)*100
            model = "Decision Tree"
        y_pred = reg.predict(X_test)
    elif method == "ml3":
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, random_state=1)
        if model_name == "knnclf":
            akurasi = round(KNNclf(X_train,y_train),4)*100
            model = " K-Nearest Neighbors"
        elif model_name == "mlpclf":
            akurasi = round(MLPclf(X_train,y_train),4)*100
            model = "Multi-layer Perceptron"
        elif model_name == "decclf":
            akurasi = round(DecisionTclf(X_train,y_train),4)*100
            model = " Decision Tree"
        elif model_name == "svmclf":
            akurasi = round(SVMclf(X_train,y_train),4)*100
            model = "Support Vector Machine"
        elif model_name == "ldaclf":
            akurasi = round(LDAclf(X_train,y_train),4)*100
            model = "Linear Discriminant Analysis"
        elif model_name == "lrclf":
            akurasi = round(LRclf(X_train,y_train),4)*100
            model = "Logistic Regression"
        elif model_name == "ridgeclf":
            akurasi = round(Ridgeclf(X_train,y_train),4)*100
            model = "Ridge Classifier"
        elif model_name == "gradclf":
            akurasi = round(Gradboostclf(X_train,y_train),4)*100
            model = "Gradient Boosting"
        elif model_name == "nbclf":
            akurasi = round(GaussianNBclf(X_train,y_train),4)*100
            model = "Gaussian Naive Bayes (NB)"
        y_pred = clf.predict(X_test)
    return (model, y_pred, y_test, akurasi)

#If packed and template and static folders in different path (maybe? dunno)
if getattr(sys, 'frozen', False):

    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')

    app = Flask(__name__, template_folder=template_folder,
                static_folder=static_folder)

else:

    app = Flask(__name__)
#Flask framework configuration
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'resources')
SECRET_KEY = 'estehmanisenakbanget'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY


@app.route("/", methods=['GET', 'POST'])
def load():
    if request.method == 'POST':
        if 'inputCSV' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['inputCSV']
        name = request.form['inputName']
        global username
        username = name
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if name == '':
            flash('Your name cannot be empty')
            return redirect(request.url)
        if (not allowed_file(file.filename)):
            flash('Your data has unknown format')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            global df
            df = pd.read_csv(file)
            global variable_selection,model_selection
            variable_selection = False
            model_selection = False
            return redirect(url_for('type_variable'))

    return render_template('load-file.html', 
                            menu="load")


@app.route("/type", methods=['GET', 'POST'])
def type_variable():
    if (df.empty):
        flash("You have to load your data first")
        return render_template('variable_type.html',
                                menu="type",
                                namamenu="Variable Types",
                                user=username, 
                                empty=True)
    else:
        if request.method == 'POST':
            numeric = request.form.getlist('examplebox')
            categorical = request.form.getlist('catexamplebox')
            date = request.form.getlist('dateexamplebox')
            global num_var
            global cat_var
            global date_var
            num_var = numeric
            cat_var = categorical
            date_var = date
            try :
                df[num_var] = df[num_var].astype('float')
                df[cat_var] = df[cat_var].astype('object')
                df[date_var] = df[date_var].astype('datetime64')
            except ValueError :
                flash("Invalid variable type")
                return redirect(request.url)
            return redirect(url_for('null_value'))
        variables = df.columns.to_list()
        return render_template('variable_type.html',
                                menu="type",
                                namamenu="Variable Types",
                                variables=json.dumps(variables),
                                user=username)


@app.route("/null", methods=['GET', 'POST'])
def null_value():
    global df
    if (df.empty):
        flash("You have to load your data first")
        return render_template('missing_value.html',
                                menu="null",
                                namamenu="Missing Values",
                                user=username,
                                empty=True)
    else:
        if request.method == 'POST':
            action = request.form.get('nullRadios')
            #global df
            if (action == "dropna"):
                df = dropvalue(df)
            elif(action == "simple"):
                df = simpleimputer(df)
            elif(action == "knn"):
                df = knnimputer(df)
            elif(action == "mice"):
                df = miceimputer(df)
            else:
                df = keeporigin(df)
                if df.isnull().values.any():
                    flash('Your data contains null values!')
                    return(redirect(request.url))
            return redirect(url_for('stats'))
        variables = df.columns.to_list()
        global df_null
        df_null = missingvalues(df)
        null_stats = [df_null.to_html(border=0,
                                    classes='table table-bordered',
                                    header="true")]
        return render_template('missing_value.html',
                                menu="null",
                                namamenu="Missing Values",
                                variables=json.dumps(variables),
                                null=null_stats,
                                user=username)

@app.route("/stats", methods=['GET', 'POST'])
def stats():
    if (df.empty):
        flash("You have to load your data first")
        return render_template('desc-stats.html',
                                menu="stats",
                                namamenu="Descriptive Statistic",
                                user=username,
                                empty=True)
    else:
        global n_data
        n_data = len(df)
        desc_stats = [df.describe().to_html(border=0,
                                            classes='table table-bordered',
                                            header="true")]
        return render_template('desc-stats.html',
                                menu="stats",
                                namamenu="Descriptive Statistic",
                                desc=desc_stats,
                                user=username)


@app.route("/box", methods=['GET', 'POST'])
def box():
    if (df.empty):
        flash("You have to load your data first")
        return render_template('boxplot.html',
                                menu="box",
                                namamenu="Box Plot",
                                user=username,
                                empty=True)
    else:
        variables = num_var
        if request.method == 'POST':
            variable = request.form.get('exampleRadios')
        else:
            variable = variables[0]

        if (len(cat_var) == 1):
            labels = np.unique(df[cat_var[0]].values)
            box = []
            for label in labels:
                y = df.loc[df[cat_var[0]] == label][variable].values.tolist()
                box.append(dict(y=y, type='box', name=label))
        else:
            y = df[variable].values.tolist()
            box = [dict(y=y, type='box', name=variable)]

        return render_template("boxplot.html",
                                menu="box",
                                namamenu="Box Plot",
                                variables=json.dumps(variables),
                                data=box,
                                vars=variable,
                                user=username)


@app.route("/hist", methods=['GET', 'POST'])
def hist():
    if (df.empty):
        flash("You have to load your data first")
        return render_template("histogram.html",
                                menu="hist",
                                namamenu="Histogram",
                                user=username,
                                empty=True)
    else:
        global variables,variable
        variables = num_var
        if request.method == 'POST':
            variable = request.form.get('exampleRadios')
        else:
            variable = variables[0]
        return render_template("histogram.html",
                                menu="hist",
                                namamenu="Histogram",
                                variables=json.dumps(variables),
                                data=json.dumps(df[variable].to_list()),
                                vars=variable, user=username)


@app.route("/pair")
def pair():
    if (df.empty):
        flash("You have to load your data first")
        return render_template("pairplot.html",
                                menu="pair",
                                namamenu="Pair Plot",
                                user=username,
                                empty=True)
    else:
        df2 = df.drop(cat_var, axis=1)
        df2 = df2.drop(date_var, axis=1)
        dim = []
        colorscale_pl = [
            [0.0, '#1cc88a'],
            [0.2, '#4e73df'],
            [0.4, '#36b9cc'],
            [0.6, '#f6c23e'],
            [0.8, '#e74a3b'],
            [1, '#858796']
        ]

        if (len(cat_var) != 0):
            encoder = OrdinalEncoder()
            vals = df[cat_var].values.reshape(-1, 1)
            label = encoder.fit_transform(vals).reshape(1, -1).tolist()[0]
        else:
            label = [0]*len(df)
        mark = dict(color=label,
                    showscale=False,
                    colorscale=colorscale_pl,
                    size=4,
                    line_color='white',
                    line_width=0.5)
        for i in range(len(df2.columns)):
            temp = dict(label=df2.columns[i],
                        values=df2[df2.columns[i]].to_list())
            dim.append(temp)
        return render_template('pairplot.html',
                                menu="pair",
                                namamenu="Pair Plot",
                                user=username,
                                data=dim,
                                mark=mark)


@app.route("/scatter", methods=['GET', 'POST'])
def scatter():
    if (df.empty):
        flash("You have to load your data first")
        return render_template("scatter.html",
                                menu="scatter",
                                namamenu="Scatter Plot",
                                user=username,
                                empty=True)
    else:
        df2 = df.drop(cat_var, axis=1)
        df2 = df2.drop(date_var, axis=1)
        variables = num_var
        if request.method == 'POST':
            xvar = request.form.get('exampleRadios1')
            yvar = request.form.get('exampleRadios2')
        else:
            xvar = variables[0]
            yvar = variables[0]
        return render_template("scatter.html",
                                menu="scatter",
                                namamenu="Scatter Plot",
                                variables=json.dumps(variables),
                                xvar=xvar,
                                yvar=yvar,
                                datax=json.dumps(df2[xvar].to_list()),
                                datay=json.dumps(df2[yvar].to_list()),
                                user=username)


@app.route("/heat")
def heat():
    if (df.empty):
        flash("Load your data first!")
        return render_template("heatmap.html",
                                menu="heat",
                                namamenu = "Heat Map",
                                user=username,
                                empty=True)
    else:
        df2 = df.drop(cat_var, axis=1)
        df2 = df2.drop(date_var, axis=1)
        label = df2.columns.to_list()
        z = df2.corr().values.tolist()
        for i in range(len(df2.columns)):
            for j in range(len(df2.columns)):
                z[i][j] = round(z[i][j], 2)
        return render_template('heatmap.html',
                                menu="heat",
                                namamenu = "Heat Map",
                                user=username,
                                zval=z,
                                label=label)


@app.route("/var", methods=['GET', 'POST'])
def var():
    if (df.empty):
        flash("You have to load your data first")
        return render_template("variable_selection.html",
                                menu="var",
                                namamenu = "Variable Selection",
                                user=username,
                                empty=True)
    else:
        if request.method == 'POST':
            global datax, datay, kind, n_clusters, xvar, yvar, cluster_var
            global mlp_score,knn_score,ridge_score,bayes_score,svm_score,dec_score,lr_score,lda_score,nb_score,gboost_score,randforest_score
            global dim,mark,uploaded_csv
            
            xvar = request.form.getlist('examplebox')
            yvar = request.form.get('exampleRadios')
            meth = request.form.get('mlRadios')
            if (len(xvar) != 0):
                cluster_var = xvar
            if (yvar in xvar):
                flash("Same variable on predictor and response!")
                return redirect(request.url)
            if ("object" in df[xvar].dtypes.values):
                flash("Your predictor variables contains non-numerical values.")
                return redirect(request.url)

            if (len(xvar) == 0):
                clusters = request.form['cluster']
                n_clusters = int(clusters)
            else:
                datax = df[xvar]
                datay = df[yvar]
                kind = meth
            global variable_selection
            variable_selection = True
            uploaded_csv = False
            if (kind == "ml1"):
                if (df[yvar].dtypes == "object"):
                    flash ('Could not convert string to float. Please check your variable type!')
                    return redirect(request.url)
                mlp_score = round(MLPregr(datax, datay),4)*100
                print("MLP Finished")
                randforest_score = round(RandomforestReg(datax,datay),4)*100
                print("Rand Forest Finished")
                gboost_score = round(GradboostReg(datax,datay),4)*100
                print("GBOST finished")
                ridge_score = round(ridgeregr(datax, datay),4)*100
                print("Ridge Finished")
                bayes_score = round(BayesianRregr(datax, datay),4)*100
                print("Bayes Finished")
                svm_score = round(SVMregr(datax, datay),4)*100
                print("SVM Finished")
                knn_score = round(KNNregr(datax, datay),4)*100
                print("KNN Finished")
                dec_score = round(DecisionTregr(datax, datay),4)*100
                print("Tree Finished")
                return redirect(url_for('model'))
            elif (kind == "ml3"):
                datay = datay.astype('category')
                knn_score = round(KNNclf(datax, datay),4)*100
                print("KNN Finished")
                mlp_score = round(MLPclf(datax, datay),4)*100
                print("MLP Finished")
                dec_score = round(DecisionTclf(datax, datay),4)*100
                print("Tree Finished")
                svm_score = round(SVMclf(datax, datay),4)*100
                print("SVM Finished")
                lda_score = round(LDAclf(datax, datay),4)*100
                print("LDA Finished")
                lr_score = round(LRclf(datax, datay),4)*100
                print("LR Finished")
                ridge_score = round(Ridgeclf(datax, datay),4)*100
                print("Ridge Finished")
                gboost_score = round(Gradboostclf(datax, datay),4)*100
                print("GBOST Finished")
                nb_score = round(GaussianNBclf(datax, datay),4)*100
                print("NB Finished")
                return redirect(url_for('model'))
            else:
                global clu,dim,mark,elbow

                for i in range (1,11):
                    model = KMeans(n_clusters=i)
                    model.fit(datax)
                    elbow[i-1] = model.inertia_
                kmeans = KMeans(n_clusters=n_clusters)
                scaler = StandardScaler()
                clu = make_pipeline(scaler, kmeans)
                clu.fit(datax)
                label = clu.predict(datax)
                dim = []
                colorscale_pl = [
                    [0.0, '#1cc88a'],
                    [0.2, '#4e73df'],
                    [0.4, '#36b9cc'],
                    [0.6, '#f6c23e'],
                    [0.8, '#e74a3b'],
                    [1, '#858796']
                ]
                mark = dict(color=label.tolist(),
                            showscale=False,
                            colorscale=colorscale_pl,
                            line=dict(color='white', width=0.5))
                for i in range(len(datax.columns)):
                    temp = dict(label=datax.columns[i],
                                values=datax[datax.columns[i]].to_list())
                    dim.append(temp)
            return redirect(url_for('model'))
        variables = df.columns.to_list()
        return render_template('variable_selection.html',
                                menu="var",
                                namamenu = "Variable Selection",
                                variables=json.dumps(variables),
                                user=username)


@app.route("/model", methods=['GET', 'POST'])
def model():
    if (df.empty):
        flash("You have to load your data first")
        return render_template('model_selection.html',
                                menu="model",
                                namamenu = "Model Selection",
                                user=username,
                                empty=True)
    else:
        if (not variable_selection):
            flash("You have to select your variables first")
            return render_template('model_selection.html',
                                    menu="model",
                                    namamenu = "Model Selection",
                                    user=username,
                                    empty=True)
        if request.method == 'POST':
            global method,model,y_pred,y_test,acc,n_data,matrix,xlabel,ylabel
            getmethod = request.form.get('exampleRadios')
            method = getmethod
            global model_selection
            model_selection = True
            if (kind != "ml2"):
                metode = method
            else:
                metode = "kmeans"
            n_data = len(datay)
            if kind == "ml1":
                model, y_pred, y_test, acc = model_performance(datax, datay, kind, metode, reg, clf)
                y_pred = y_pred.tolist()
                y_test = y_test.tolist()
            elif kind == "ml3":
                model, y_pred, y_test, acc = model_performance(
                    datax, datay, kind, metode, reg, clf)
                matrix = confusion_matrix(y_test,y_pred).tolist()
                xlabel=np.unique(y_pred).tolist()
                ylabel=np.unique(y_test).tolist()
            else:
                print(clu)
                #Nothing
            return redirect(url_for('perf'))
        
        if kind == "ml1":
            return render_template('model_selection.html',
                                    menu="model",
                                    namamenu = "Model Selection",
                                    user=username,
                                    mlp=mlp_score,
                                    randfor=randforest_score,
                                    gboost=gboost_score,
                                    ridge=ridge_score,
                                    bayes=bayes_score,
                                    svm=svm_score,
                                    knn=knn_score,
                                    dec=dec_score,
                                    kind=kind)
        elif kind == "ml3":
            return render_template('model_selection.html',
                                    menu="model",
                                    namamenu = "Model Selection",
                                    user=username,
                                    mlp=mlp_score,
                                    gboost=gboost_score,
                                    ridge=ridge_score,
                                    svm=svm_score,
                                    knn=knn_score,
                                    dec=dec_score,
                                    lda=lda_score,
                                    lr=lr_score,
                                    naivebayes=nb_score,
                                    kind=kind)
        else:
            return render_template('model_selection.html',
                                    menu="model",
                                    namamenu = "Model Selection",
                                    user=username,
                                    data=dim,
                                    mark=mark,
                                    kind=kind,
                                    elbow=elbow)


@app.route("/perf", methods=['GET', 'POST'])
def perf():
    if (df.empty):
        flash("You have to load your data first")
        return render_template('model_performance.html',
                                menu="perf",
                                namamenu="Model Performance",
                                user=username,
                                empty=True)
    else:
        print(reg)
        print(clf)
        if (not model_selection):
            flash("You need to select your model first.")
            return render_template('model_performance.html',
                                    menu="perf",
                                    namamenu="Model Performance",
                                    user=username,
                                    empty=True)
        if kind == "ml1":
            return render_template('model_performance.html',
                                    menu="perf",
                                    namamenu="Model Performance",
                                    user=username,
                                    model=model,
                                    y_pred=y_pred,
                                    y_test=y_test,
                                    acc=acc,
                                    n_data=n_data,
                                    jenis=kind)
        elif kind == "ml3":
            return render_template('model_performance.html',
                                    menu="perf",
                                    namamenu="Model Performance",
                                    user=username,
                                    model=model,
                                    matrix=matrix,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    acc=acc,
                                    n_data=n_data,
                                    jenis=kind)
        else:
            return render_template('model_performance.html',
                                    menu="perf",
                                    namamenu="Model Performance",
                                    user=username,
                                    model="K-Means Clustering",
                                    acc="NA",
                                    n_data=n_data,
                                    jenis=kind,
                                    data=dim,
                                    mark=mark)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if (df.empty):
        flash("You have to load your data first")
        return render_template('predict_data.html',
                                menu="predict",
                                namamenu="Predict New Data",
                                user=username,
                                empty=True)
    if (not model_selection):
        flash("You have to train your model first")
        return render_template('predict_data.html',
                                menu="predict",
                                namamenu="Predict New Data",
                                user=username,
                                empty=True)
    else:
        
        global uploaded_csv
        if request.method == 'POST':
            if 'predictCSV' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['predictCSV']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                global df_predict
                global df_results
                df_predict = pd.read_csv(file)
                uploaded_csv = True
                if (kind=="ml1"):
                    new_column = reg.predict(df_predict[xvar])
                    df_results = pd.DataFrame(df_predict[xvar])
                    df_results[yvar] = new_column
                elif (kind=="ml3"):
                    new_column = clf.predict(df_predict[xvar])
                    df_results = pd.DataFrame(df_predict[xvar])
                    df_results[yvar] = new_column
                else:
                    global df_clustering,labels,cluster_var
                    df_clustering = df.append(df_predict,ignore_index=True)
                    datacluster = df_clustering[cluster_var]
                    clu.fit(datacluster)
                    labels = clu.predict(datacluster)
                    df_clustering['Label'] = labels
                    df_results = df_clustering.copy()
                global desc
                desc = [df_predict.describe().to_html(border=0,
                                            classes='table table-bordered',
                                            header="true")]
                return render_template('predict_data.html',
                                        menu="predict",
                                        namamenu="Prediction",
                                        user=username,
                                        upload=uploaded_csv, 
                                        kind=kind, 
                                        desc=desc,
                                        modelstatus=model_selection)
        else:
            return render_template('predict_data.html',
                                    upload=uploaded_csv,
                                    menu="predict",
                                    namamenu="Prediction",
                                    user=username, 
                                    kind=kind, 
                                    desc=desc,
                                    modelstatus=model_selection)

@app.route('/about')
def about():
    return render_template('about.html',
                            menu="about",
                            user=username)

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plothist.png')
def plot_pnghist():
    fig = create_figurehist()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plotreg.png')
def plot_reg():
    fig = create_regresults()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plotclf.png')
def plot_clf():
    fig = create_clfresults()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plotclu.png')
def plot_clu():
    fig = create_cluresults()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/downloadcsv')  
def download_csv():  
    csv = df_results.to_csv(index=False)  
    response = make_response(csv)
    cd = 'attachment; filename=results.csv'
    response.headers['Content-Disposition'] = cd 
    response.mimetype='text/csv'
    return response

if __name__ == "__main__":
    try:
        app.run()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
