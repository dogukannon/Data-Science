import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=sns.load_dataset("titanic")
""""
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df.dtypes
"""

#converting boolean variables to integer
for col in df.columns:
    if df[col].dtypes =="bool":
       df[col] = df[col].astype(int)

def check_df(dataframe, number=5):
    print("############### Shape ###############")
    print(dataframe.shape)
    print("############### Types ###############")
    print(dataframe.dtypes)
    print("############### Head ###############")
    print(dataframe.head(number))
    print("############### Tail ###############")
    print(dataframe.tail(number))
    print("############### NA ###############")
    print(df.isnull().sum())
    print("############### Quartiles ###############")
    print(df.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

"""
#categoric variables
categoric_col = [col for col in df.columns if str(df[col].dtypes) in ["object","category","bool"]]

#numeric but coded categoric
num_but_categoric = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ["int64", "float64"]]

#categorical variables, but the number of classes is very high and the cardinality is high. most probably they do not carry important information
categoric_but_cardinal = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category"]]

#merging all categorical columns
cat_cols = categoric_col + num_but_categoric

#drop those that seem categorical but are not
cat_cols = [col for col in cat_cols if col not in categoric_but_cardinal]


#selecting numerical variables
num_cols =[col for col in df.columns if str(df[col].dtypes) in ["float64","int64"]]
#checking variable coded numerical but it is like categorical variable
num_cols= [col for col in num_cols if col not in cat_cols]
"""

def grab_col_names(dataframe, cat_th=10, car_th=20):
   """
   Lists the categorical, numeric and categorical but cardinal variables in the data set.

   Parameters
   ----------
   dataframe: dataframe
        working dataframe.
   cat_th: int, float
        unique value threshold of variables that are numeric but behave like categorical variables
   car_th: int, float
        unique value threshold of variables that are categorical but also they are cardinal variables

   Returns
   -------
   cat_cols: list
        categorical variables list
   num_cols: list
        numeric variables list
   cat_but_car: list
        categorical variables but they are also cardinal variables list

   Notes
   -------
   total number of variables = cat_cols + num_cols + cat_but_cols

   """

   cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object","category","bool"]]
   num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and str(dataframe[col].dtypes) in ["float64","int64","int32"]]
   cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["object", "category"]]
   cat_cols = cat_cols +num_but_cat
   cat_cols = [col for col in cat_cols if col not in cat_but_car]
   num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["float64","int64"]]
   num_cols = [col for col in num_cols if col not in cat_cols]

   print(f"Observations:{dataframe.shape[0]}")
   print(f"Variables:{dataframe.shape[1]}")
   print(f"cat_cols:{len(cat_cols)}")
   print(f"num_cols:{len(num_cols)}")
   print(f"cat_but_car:{len(cat_but_car)}")
   print(f"num_but_car:{len(num_but_cat)}")

   return cat_cols,num_cols,cat_but_car


####### Categorical variable's data analysis #######
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))
    print("####################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

################################################################

###### Numerical variable's data analysis #######

def num_summary(dataframe, numerical_col, plot=False):
    quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("########## "+numerical_col.upper()+" ##########")
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in cat_cols:
    if df[col].dtypes == bool:
        print("boolean can not plotting with countplot function")
        df[col]=df[col].astype(int)
        print("boolean converted to int")
        cat_summary(df, col, plot=True)
        print("####################################")

    else:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df,col,plot=True)



#####Target variable analysis#####


#analysis of target variable with categorical variables
def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"TARGET MEAN":dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"survived",col)

#analysis of target variable with numeric variables
def target_summary_with_num(dataframe,target,numeric_col):
    print(dataframe.groupby(target).agg({numeric_col:"mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"survived",col)
################################################################

################################# Analysis of correlation ####################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df1 = pd.read_csv("datasets/breast_cancer.csv")
#dropping unused columns
df1= df1.iloc[:, 1:-1]

cat_cols1, num_cols1, cat_but_car1 = grab_col_names(df1)
"""
corr= df1[num_cols1].corr()

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr,cmap="RdBu")
plt.show()
"""
##########Deleting high correlation variables##########
#if we have high correlation between two variables we can optionally can delete one of these. Because other variable may represent deleted variable


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix =corr.abs()

    #converting upper triangle matrix
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    upper_triangle_matrix

    #selecting high correlated variables
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df1[num_cols1])
high_correlated_cols(df1.drop(drop_list, axis=1),plot=True)