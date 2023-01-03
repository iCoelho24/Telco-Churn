import matplotlib.pyplot as plt
import pandas as pd
from IPython.display            import Markdown
from sklearn.preprocessing      import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import sklearn.metrics as metrics

def plot_roc_curve(X_teste ,y_teste, model):
    roc_model = model.predict_proba(X_teste)
    roc_model = roc_model[:, 1]
    fper, tper, thresholds = roc_curve(y_teste, roc_model)
    auc = metrics.roc_auc_score(y_teste, roc_model)
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='AUC ='+str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc=4)
    plt.show()

def func_LabelEncoder(df):
    return LabelEncoder().fit_transform(df)

def LazzyLabelEncoder(df, text=True):
    text_LazzyLabelEncoder = ''
    dicio_LazzyLabelEncoder = {}
    for i, j in enumerate(df.unique()):
        text_LazzyLabelEncoder += f'{i} - {j}\n '
        dicio_LazzyLabelEncoder[j] = i
    return text_LazzyLabelEncoder[:-2] if text==True else dicio_LazzyLabelEncoder

def df_informations(df):
    df_info = pd.DataFrame({'Not Null': df.notnull().count(),
                'Null': df.isnull().sum(),
                'Perce Null': df.isnull().mean(),
                'Unique': df.nunique(),
                'Dtype': df.dtypes
                })

    df_dtype = pd.DataFrame(df_info['Dtype'].value_counts())
    df_dtype['Perce'] = round(df_dtype['Dtype'] / df_dtype['Dtype'].sum(), 2)

    text = f'Dataset has {df.shape[0]} rows and {df.shape[1]} columns. From these, we have:'

    df_info = df_info.style.background_gradient(cmap='jet', subset=['Perce Null']).format({'Perce Null': '{:.2%}'})
    df_dtype = df_dtype.style.background_gradient(cmap='YlGn', subset=['Perce']).format({'Perce': '{:.2%}'})

    display(Markdown("<H3 style='text-align:left;float:lfet;'>Information about the Dataset"))
    display(Markdown(f'<H5> {text}'))
    display(df_info)
    display(Markdown("<H3 style='text-align:left;float:lfet;'>About Dtypes we have:"))
    display(df_dtype)