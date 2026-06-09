# %% [markdown]
# # Martí Puig i Sebastián Luna Competició AA1

# %% [markdown]
# ## Importació de Llibreries

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from typing import Iterator
import re #per detectar regular expressions
import warnings
warnings.filterwarnings("ignore")
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# Set the precision of the display to 3 decimal places
pd.set_option('display.precision', 3)

# %%
#Constants
SEED:int = 383006
N_COMP:float = 0.99 #pel PCA
SEQ_CTRL_SCALED:bool = True #Per escalar seq_ctrl com a cosinus

# %% [markdown]
# ## Anàlisi Exploratori de les Dades (AED)

# %%
csi = pd.read_csv("train_nt.csv",header=0,delimiter=";")
csi_final_test = pd.read_csv("test_nolabels_nt.csv",header=0,delimiter=";") #es el dataset on intentarem predir els labels pel Kaggle

# %%
csi.columns #noms dels atributs, recordem que ID NO es una feature, aixi que la treurem fora

# %%
csi = csi.drop(columns="ID")
csi

# %% [markdown]
# Explicació de les Variables:
# - `seq_ctrl`: uint16 seqüència de nombres per identificar el csi. 
# - `aoa`: float32, attribut  sintétic generat al post-processament, representa l'angle estimat d'arribada (azimuth, en radians) calculat desde la fase de diférencia entre 2 antenas receptoras
# - `rssi1`: int8, indicador de l'intensitat del senyal mesurat rebuda per la primera antena. 
# - `rssi2`: int8, indicador de l'intensitat del senyal mesurat rebuda per la segona antena.  
# - Dades Raw CSI en forma de 64 nombres complexos, primer per la antena 1, segon per la antena 2, de la forma següent:
# 
# I0_1 Q0_1 I1_1 Q1_1 ... I63_1 Q63_1
# 
# I0_2 Q0_2 ... I63_2 Q63_2 
# 
# on In_X: Valor In-phase del subcarrier n-èssim $(0 \leq n \leq 63)$ a la antena X (X = 1 o 2) 
# i Qn_X: Valor Quadrature del subcarrier n-èssim a la antena X
# - `position` (label): int8, valor entre 0 i 9, corresponent a la posició del target device (Variable Target)

# %%
csi_shape = csi.shape
print(f"El dataset té {csi_shape[0]} mostres i {csi_shape[1]} attributs")

# %%
csi.describe()

# %%
csi['seq_ctrl'].value_counts() #veiem que hi han sequencies que es repeteixen

# %%
csi['position'].value_counts() #el nombre de mostres a cada posicio es similar

# %%
sn.pairplot(data=csi[["seq_ctrl","aoa","rssi1","rssi2","position"]], hue='position',palette="coolwarm") #no seleccionem ara les RAW CSI variables

# %% [markdown]
# Veiem que potser podriem treure rssi1 o rssi2 ja que correlan linealment, ho decidirem al apartat de Dataset Cleaning

# %% [markdown]
# ## Dataset Cleaning
# - Eliminar Outliers
# - Reduir Variables
# - Missing Values
# - Afegir o Modificar Variables
# - Escalar Dades
# - Balancejar Dataset (si no ho está)

# %% [markdown]
# ### Eliminar Outliers

# %%
#Histograma i Boxplot de rssi1
fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios':[1,4]}, figsize=(9,5))
csi.boxplot(column = "rssi1", ax=axes[0])
csi.hist(column = "rssi1", ax = axes[1])

# %%
#Histograma i Boxplot de rssi2
fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios':[1,4]}, figsize=(9,5))
csi.boxplot(column = "rssi2", ax=axes[0])
csi.hist(column = "rssi2", ax = axes[1])

# %%
#Histograma i Boxplot de AOA
fig, axes= plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(9,5))
csi.boxplot(column='aoa',ax=axes[0])
csi.hist(column='aoa', ax=axes[1])

# %%
#AOA es l'angle en graus multiplicat per 100, sabem que va de -90º a 90º
csi.aoa.max()

# %%
csi.aoa.min()

# %% [markdown]
# Els valors minims y maxims d'aoa no sorten del interval $[-90º,90º]$.
# 
# Semblaria que no hi han outliers en aquestes variables.
# 
# Estudiem les dades RAW CSI:

# %% [markdown]
# ### Eliminar Outliers i Variables de RAW CSI

# %%
# Fem una cerca per veure si n'hi han variables que son sempre 0 i les emmagatzemem a carriers_0
carriers_0:list[str] = []
for i in (1,2):
    for j in range(64):
        subcarrier_i:str = f"I{j}_{i}"
        subcarrier_q:str = f"Q{j}_{i}"
        if csi[subcarrier_i].max() == np.float64(0.0) and csi[subcarrier_i].max() == np.float64(0.0):
            carriers_0.append(subcarrier_i)
        if csi[subcarrier_q].max() == np.float64(0.0) and csi[subcarrier_q].max() == np.float64(0.0):
            carriers_0.append(subcarrier_q)
print(carriers_0)

# %%
len(carriers_0)

# %% [markdown]
# Aquestes variables son sempre 0, les podem treure perquè no aportan informació.
# 

# %%
csi_filtered = csi.drop(columns=carriers_0)
csi_uncorr = csi.drop(columns=carriers_0)
csi_filter_plus = csi.drop(columns=carriers_0)
csi_pca = csi.drop(columns=carriers_0)
csi_angle_pca = csi.drop(columns=carriers_0)
csi_filter_plus_uncorr = csi.drop(columns=carriers_0)
csi_filtered.describe()

# %% [markdown]
# # Els següents dataset tretes les I's i Q's 0:
# - csi_filtered: Cap canvi adicional
# - csi_uncorr: Es converteixen a Modul i Angle + Aplicacio de uncorr_vars()
# - csi_filter_plus: S'aplica directament min_mutual_info()
# - csi_pca: Aplicat PCA
# - csi_angle_pca: Modul i Angle + PCA
# - csi_filtered_plus_uncorr

# %% [markdown]
# Segons la informació que hem trobat sobre aquestes dades d'[OFDM]("https://www.cwnp.com/understanding-ofdm-part-2-2/"):
# 
# Es tracta de nombres complexos de la forma $H= I + iQ$, on $i^2=-1$.
# 
# Així que podriem sustituir les variables I, Q per A (módul) i Theta (angle).
# 
# Anem a crear una serie de funcions per processars els datasets, com per eliminar variables que donen poca informació sobre la resposta, que estiguin correlades entre ellas o per transfromar les dades:

# %%
def has_attribute(df:pd.DataFrame,attr:str)->bool:
    """Devuelve True si df[attr] existe"""
    try:
        result = df[attr]
        return True
    except:
        return False



def i_q_indexs(i_q_list:list[str])->list[tuple[int,int]]:
    """Retorna una llista de tuples (j,k), on Ij_k o Qj_k"""
    parelles:list[tuple[int,int]] = [] #lo la incialitzem de la mateixa mida que i_q_list per si hi han attributs que no son I o Q a la llista
    for elem in i_q_list:
        match = re.findall(r"[IQ](\d+)_(\d+)", elem)
        try:
            parelles.append((int(match[0][0]),int(match[0][1])))
        except:
            continue
    return parelles




def find_i_q(df:pd.DataFrame)->list[str]:
    """Retorna una llista amb el nom de les columnes de la forma Ix_y o Qx_y"""
    cols:list[str] = [column for column in df.columns]
    i_qs:list[str] = []
    for i in range(len(cols)):
        columna = cols[i]
        match=re.search(r"[IQ](\d+)_(\d+)", columna)
        if match is not None:
            i_qs.append(columna)
    return i_qs



def find_raw_csi_format(df:pd.DataFrame,lletra:str,num:int=-1)->list[str]:
    """Retorna una llista amb el nom de les columnes de la forma lletrax_y o lletrax_num (si es dona num >= 0)"""
    cols:list[str] = [column for column in df.columns]
    troballes:list[str] = []
    for i in range(len(cols)):
        columna = cols[i]
        if num > 0:
            match=re.search(rf"{lletra}(\d+)_{num}", columna)
        else:
            match=re.search(rf"{lletra}(\d+)_(\d+)", columna)
        if match is not None:
            troballes.append(columna)
    return troballes




def complex_conversion(df:pd.DataFrame)->None:
    """
    Transforma les dades I i Q a A (modul) i O (angle).

    Avis: Modifica el dataset donat com a parametre.

    Prec: Cal previament eliminar les I's i Q's que siguin 0
    """
    rows:int = df.shape[0]
    i_q_iter:list[tuple[int,int]] = i_q_indexs(find_i_q(df))
    for tup in i_q_iter:
        j:int = tup[0]
        k:int = tup[1]
        mod:str = f"A{j}_{k}"
        angl:str = f"O{j}_{k}"
        i_str = f"I{j}_{k}"
        q_str = f"Q{j}_{k}"
        if has_attribute(df,mod): #pel cas on hi hagi un i_q_iter que es repeteix
            continue
        #S'asegura de donar un valor a I i Q en cas de que un dels 2 no existeixi
        try:
            i_val = df[i_str]
            df.drop(columns=[i_str],inplace=True)
        except:
            i_val = np.zeros(rows)
        try:
            q_val = df[q_str]
            df.drop(columns=[q_str],inplace=True)
        except:
            q_val = np.zeros(rows)
        complex = i_val + q_val * 1j
        np.angle(complex)
        df[mod] = np.abs(complex)
        df[angl] = np.angle(complex)


# %%
def min_mutual_info(df:pd.DataFrame,min_mut_infor:float=0.1)->list[str]:
    """
    Retorna les variables que no aporten prou informació sobre la variable objectiu
    """
    X = df.drop(columns=["position"])
    y = df["position"]
    puntuacions_mi = mutual_info_classif(X, y, random_state=42)
    cols = []
    for i in range(len(puntuacions_mi)):
        if puntuacions_mi[i] < min_mut_infor:
            cols.append(X.columns[i])
    return cols

# %%
#Algorisme que funciona com Sieve d' Eratosthenes pero per descartar les variables
def uncorr_vars(df:pd.DataFrame,vars:list[str],min_corr:float=0.6, drop_out:bool=False)->list[str]: #O(n_vars**2 * rows(df))
    """Devuelve las variables que poseen una abs(correlacion) < min_corr.
    
    :param: drop_out: si es True, entonces devuelve las variables correlacionadas que se tendrian que eliminar
    """
    n_vars:int = len(vars)
    vars_select:list[bool] = [True for _ in range(n_vars)]
    for i in range(n_vars): #Bucle que calcula la correlacio entre variables i selecciona les 'no correlades' (corr < min_corr)
        var_i = vars[i]
        if vars_select[i]:
            for j in range(i+1,n_vars):
                if vars_select[j]:
                    var_j = vars[j]
                    corr = df[var_i].corr(df[var_j])
                    vars_select[j] = np.abs(corr) < min_corr
    if drop_out:
        return [vars[j] for j in range(n_vars) if not(vars_select[j])]
    return [vars[j] for j in range(n_vars) if vars_select[j]]

# %%
#csi_uncorr
complex_conversion(csi_uncorr)
moduls_1 = find_raw_csi_format(csi_uncorr,"A",1)
uncorr_moduls_1 = uncorr_vars(csi_uncorr,moduls_1[1:]) #de 52 variables incialment ens quedem amb 5
moduls_2 = find_raw_csi_format(csi_uncorr,"A",2)
uncorr_moduls_2 = uncorr_vars(csi_uncorr,moduls_2) # de 52 variables incialment ens quedem amb 12
uncorr_moduls = uncorr_vars(csi_uncorr,uncorr_moduls_1 + uncorr_moduls_2)
angl_1 = find_raw_csi_format(csi_uncorr,"O",1)
angl_2 = find_raw_csi_format(csi_uncorr,"O",2)
uncorr_angl_1 = uncorr_vars(csi_uncorr,angl_1) # de 52 vars pasem a 2
uncorr_angl_2 = uncorr_vars(csi_uncorr,angl_2) # de 52 vars pasem a 1
uncorr_raw_csi = uncorr_moduls + uncorr_angl_1 + uncorr_angl_2
vars_to_drop_uncorr = uncorr_vars(csi_uncorr,moduls_1[1:] + moduls_2 + angl_1 + angl_2,drop_out=True) #fem el drop de les variables no correlades
csi_uncorr.drop(columns=vars_to_drop_uncorr,inplace=True)
csi_uncorr.describe()

# %%
#csi_filter_plus
vars_to_drop_filter_plus = min_mutual_info(csi_filter_plus)
csi_filter_plus.drop(columns=vars_to_drop_filter_plus, inplace=True)
csi_filter_plus.describe()

# %%
#csi_pca
X_pca = csi_pca.loc[:,csi_pca.columns != 'position']
y_pca = csi_pca['position']
pca = PCA(n_components=N_COMP)
X_pca = pca.fit_transform(X_pca)
csi_pca = pd.DataFrame(X_pca)
csi_pca['position'] = y_pca
csi_pca

# %%
#csi_angle_pca
complex_conversion(csi_angle_pca)
X_angle_pca = csi_angle_pca.loc[:,csi_angle_pca.columns != 'position']
y_angle_pca = csi_angle_pca['position']
pca = PCA(n_components=N_COMP)
X_angle_pca = pca.fit_transform(X_angle_pca)
csi_angle_pca = pd.DataFrame(X_angle_pca)
csi_angle_pca['position'] = y_angle_pca
csi_angle_pca

# %%
#csi_filtered_plus_uncorr
vars_to_drop_filtered_plus_uncorr_1 = min_mutual_info(csi_filter_plus_uncorr)
csi_filter_plus_uncorr.drop(columns=vars_to_drop_filtered_plus_uncorr_1, inplace=True)
complex_conversion(csi_filter_plus_uncorr)

a_1 = find_raw_csi_format(csi_filter_plus_uncorr,"A",1)
o_1 = find_raw_csi_format(csi_filter_plus_uncorr,"O",1)
a_2 = find_raw_csi_format(csi_filter_plus_uncorr,"A",2)
o_2 = find_raw_csi_format(csi_filter_plus_uncorr,"O",2)
vars_to_drop_filtered_plus_uncorr_2 = uncorr_vars(csi_filter_plus_uncorr,vars=a_1 + o_1 + a_2 + o_2,drop_out=True)
csi_filter_plus_uncorr.drop(columns=vars_to_drop_filtered_plus_uncorr_2,inplace=True)
csi_filter_plus_uncorr.describe()

# %% [markdown]
# ### Missing Values

# %%
csi_filtered.describe()

# %%
#Calculem el nombre de missing Values
csi_filtered.isna().sum()

# %% [markdown]
# Com Pandas no detecta missing values i no hem trobat outliers o valors extranys, concluim que no hi han missing values.

# %% [markdown]
# ### Escalar les dades

# %% [markdown]
# Primer, seq_ctrl es una seqüència de nombres per identificar el csi codificada com uint16, es a dir qu'agafa valors entre 0 i 65535. En principi, seq_ctrl no té un significat ordinal (un seq_ctrl 2000 no es més important qu'un 0) aleshores s'ha de tractar com variable categórica. Té massas categorías com per aplicar one-hot enconding, però si assumim que té propietats ciclicas (desprès d'una seqüència 65535 torna a hi haver un 0) podriem crear dues variables de la forma:
# 
# seq_ctrl_sin $=sin(2 \cdot \pi \cdot i /65535)$, amb $i=$ seq_ctrl
# 
# seq_ctrl_cos $=cos(2 \cdot \pi \cdot i /65535)$, amb $i=$ seq_ctrl
# 
# Com a segona opció, podriem considerar simplement que seq_ctrl no aporta informació sobre la posició i aleshores sería millor eliminarla.

# %%
#seq_ctrl_array = csi_filtered["seq_ctrl"] #guardem el seq_ctrl per desprès fer canvis, de moment ho eliminarem de csi_filtered

# %%
#csi_filtered = csi_filtered.drop(columns="seq_ctrl")

# %%
def scaling_preprocessing(X, scaler=None,scale_seq_ctrl:bool=False)->tuple[pd.DataFrame,MinMaxScaler]: #funcio extraida de la practica 4 Linear Regression
    """Escala los datos numericos de X y los devuelve escalados y con su escalador.
    
    Prec: X no debe tener variables categoricas ni NA's

    :param: scaler: se debe indicar el utilizado en los datos de train cuando se utilicen los de test

    :param: scale_seq_ctrl si es True realiza el escalado cos y sin
    """
    print('Original shape:{}'.format(X.shape))
    if scale_seq_ctrl:
        categorical_columns = ["seq_ctrl","seq_ctrl_sin","seq_ctrl_sin"]
        X["seq_ctrl_sin"] = np.sin(2 * np.pi * X["seq_ctrl"] / 65535)
        X["seq_ctrl_cos"] = np.cos(2 * np.pi * X["seq_ctrl"] / 65535)
        X.drop(columns=["seq_ctrl"],inplace=True)
    else:
        categorical_columns = []
    numerical_columns = [c for c in X.columns if c not in categorical_columns]
    # Solo escalamos la columna numerica
    numerical_columns = [c for c in X.columns if c not in categorical_columns]
    if scaler is None:
        # Solo creamos el scaler para entrenarse con datos de train (para que no haya fugas)
        scaler = MinMaxScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    else:
        X[numerical_columns] = scaler.transform(X[numerical_columns])

    #Aqui podriam afegir el escalat de la variable categorica

    print('New shape:{}'.format(X.shape))
    return X, scaler

# %%
def df_train_test_split(df:pd.DataFrame,test_size:float=0.25,seed:int=SEED)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Fa un train  Test Split pel Dataframe df i retorna X_train, X_test, y_train, y_test"""
    X = df.loc[:,df.columns != 'position']
    y = df['position']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=seed, stratify=csi_filtered['position']) #stratify fara que es conservin les mateixes proporcions de la feature position
    return (X_train, X_test, y_train, y_test)

# %%
#csi_filtered
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = df_train_test_split(csi_filtered)
X_train_filtered, scaler_filtered = scaling_preprocessing(X_train_filtered,scale_seq_ctrl=SEQ_CTRL_SCALED)
X_test_filtered, _ = scaling_preprocessing(X_test_filtered,scaler_filtered,scale_seq_ctrl=SEQ_CTRL_SCALED)

# %%
#csi_uncorr
X_train_uncorr, X_test_uncorr, y_train_uncorr, y_test_uncorr = df_train_test_split(csi_uncorr)
X_train_uncorr, scaler_uncorr = scaling_preprocessing(X_train_uncorr,scale_seq_ctrl=SEQ_CTRL_SCALED)
X_test_uncorr, _ = scaling_preprocessing(X_test_uncorr,scaler_uncorr,scale_seq_ctrl=SEQ_CTRL_SCALED)

# %%
#csi_filter_plus
X_train_filter_plus, X_test_filter_plus, y_train_filter_plus, y_test_filter_plus = df_train_test_split(csi_filter_plus)
X_train_filter_plus, scaler_filter_plus = scaling_preprocessing(X_train_filter_plus,scale_seq_ctrl=SEQ_CTRL_SCALED)
X_test_filter_plus, _ = scaling_preprocessing(X_test_filter_plus,scaler_filter_plus,scale_seq_ctrl=SEQ_CTRL_SCALED)

# %%
#csi_pca
X_train_pca, X_test_pca, y_train_pca, y_test_pca = df_train_test_split(csi_pca)
X_train_pca.shape

# %%
#csi_angle_pca
X_train_angle_pca, X_test_angle_pca, y_train_angle_pca, y_test_angle_pca = df_train_test_split(csi_angle_pca)
X_train_angle_pca.shape

# %%
#csi_filter_plus_uncorr
X_train_filter_plus_uncorr, X_test_filter_plus_uncorr, y_train_filter_plus_uncorr, y_test_filter_plus_uncorr = df_train_test_split(csi_filter_plus_uncorr)
X_train_filter_plus_uncorr, scaler_filter_plus_uncorr = scaling_preprocessing(X_train_filter_plus_uncorr,scale_seq_ctrl=SEQ_CTRL_SCALED)
X_test_filter_plus_uncorr, _ = scaling_preprocessing(X_test_filter_plus_uncorr,scaler_filter_plus_uncorr,scale_seq_ctrl=SEQ_CTRL_SCALED)


# %% [markdown]
# ### Balancejar el dataset
# Com que hem considerat qu'el dataset estaba balancejat perquè no hi havien grans diferencies per les classes, amb això acabem el Dataset Cleaning, ara comencarem a entrenar models.

# %% [markdown]
# ### Funcions per l'exportació a Kaggle
# 
# Es tractan de funcions per convertir les dades de test de Kaggle el mateix format dels diferents datasets i escriure en un fitxer les prediccions de la variable position

# %%
#funcions per transformar csi_final_test per que sigui com X i y (trasnformacions i reduccions de variables inclosas)
def final_test_filtered(df_raw:pd.DataFrame=csi_final_test,scaler:MinMaxScaler=scaler_filtered,columns_to_drop:list[str]=carriers_0,scale_seq_ctrl:bool=SEQ_CTRL_SCALED)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de filtered.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop,inplace=True) # fem un drop de les I's i Q's igual a 0
    X_final_test, _ = scaling_preprocessing(df,scaler,scale_seq_ctrl=scale_seq_ctrl)
    return X_final_test

# %%
def final_test_uncorr(df_raw:pd.DataFrame=csi_final_test,scaler:MinMaxScaler=scaler_uncorr,columns_to_drop:list[str]=carriers_0 + vars_to_drop_uncorr,scale_seq_ctrl:bool=SEQ_CTRL_SCALED)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de uncorr.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop[:48],inplace=True) # fem un drop de les I's i Q's igual a 0
    complex_conversion(df)
    df.drop(columns=columns_to_drop[48:],inplace=True) #treim les variable no correlades
    X_final_test, _ = scaling_preprocessing(df,scaler,scale_seq_ctrl=scale_seq_ctrl)
    return X_final_test

# %%
def final_test_filter_plus(df_raw:pd.DataFrame=csi_final_test,scaler:MinMaxScaler=scaler_filter_plus,columns_to_drop:list[str]=carriers_0 + vars_to_drop_filter_plus,
                           scale_seq_ctrl:bool=SEQ_CTRL_SCALED)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de filter_plus.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop,inplace=True)
    X_final_test, _ = scaling_preprocessing(df,scaler,scale_seq_ctrl=scale_seq_ctrl)
    return X_final_test


# %%
def final_test_pca(df_raw:pd.DataFrame=csi_final_test,columns_to_drop:list[str]=carriers_0,n_comp:float=N_COMP)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de pca.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop,inplace=True)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(df)
    return pd.DataFrame(X_pca)

# %%
def final_test_angle_pca(df_raw:pd.DataFrame=csi_final_test,columns_to_drop:list[str]=carriers_0,n_comp:float=N_COMP)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de angle_pca.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop,inplace=True)
    complex_conversion(df)
    pca = PCA(n_components=n_comp)
    X_angle_pca = pca.fit_transform(df)
    return pd.DataFrame(X_angle_pca)


# %%
def final_test_filter_plus_uncorr(df_raw:pd.DataFrame=csi_final_test,scaler:MinMaxScaler=scaler_filter_plus_uncorr,
                                  columns_to_drop:list[str]=carriers_0 + vars_to_drop_filtered_plus_uncorr_1 + vars_to_drop_filtered_plus_uncorr_2,
                                  scale_seq_ctrl:bool=SEQ_CTRL_SCALED)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de filter_plus_uncorr.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop[:183],inplace=True) # fem un drop de les I's i Q's igual a 0 i variables de min_info_mutua
    complex_conversion(df)
    df.drop(columns=columns_to_drop[183:],inplace=True) #treim les variable no correlades
    X_final_test, _ = scaling_preprocessing(df,scale_seq_ctrl=scale_seq_ctrl)
    return X_final_test

# %%
#funcions per escriure el resultat a un fitxer
def output_submission(y:np.ndarray,filename:str="out")->None:
    """Escribe y en el fichero filename.csv para la submission. No hay que incluir '.csv' en filename"""
    filename= filename + ".csv"
    with open(filename,"w") as f:
        print("ID,POSITION",file=f)
        for i in range(len(y)):
            print(f"{i},{y[i]}",file=f)
    print("Fitxer d'output generat")

# %% [markdown]
# # Seleccio de Dades per l'entrenament

# %%
def dataset_iterator(x_train:list[pd.DataFrame]=[X_train_filtered, X_train_uncorr, X_train_filter_plus, X_train_pca, X_train_angle_pca,X_train_filter_plus_uncorr],
                    x_test:list[pd.DataFrame]=[X_test_filtered, X_test_uncorr, X_test_filter_plus, X_test_pca, X_test_angle_pca,X_test_filter_plus_uncorr],
                    y_train:list[np.ndarray]=[y_train_filtered,y_train_uncorr,y_train_filter_plus, y_train_pca, y_train_angle_pca,y_train_filter_plus_uncorr],
                    y_test:list[np.ndarray]=[y_test_filtered,y_test_uncorr,y_test_filter_plus,y_test_pca,y_test_angle_pca,y_test_filter_plus_uncorr],
                    x_final_test:list[pd.DataFrame] =[final_test_filtered(),final_test_uncorr(),final_test_filter_plus(),final_test_pca(),final_test_angle_pca(),final_test_filter_plus_uncorr()],
                    noms:list[str] = ["csi_filtered","csi_uncorr","csi_filter_plus","csi_pca","csi_angle_pca","csi_filter_plus_uncorr"])->Iterator[tuple[pd.DataFrame,pd.DataFrame,np.ndarray,np.ndarray,pd.DataFrame,str]]:
    """Iterador per obtenir X_train,X_test,y_train,y_test sobre els diferents datasets"""
    num_iters:int = len(x_train)
    for i in range(num_iters):
        yield (x_train[i], x_test[i], y_train[i], y_test[i],x_final_test[i],noms[i])


# %%
select_dataset = dataset_iterator() #crea el iterador de datasets

# %%
best_datasets_result_df = pd.DataFrame(index=[], columns= ['Classifier','Accuracy', 'F1 Macro', 'Precision Macro', 'Recall Macro'])

# %%
#Cada cop que s'executa aquest bloc s'escull el seguent dataset de la llista, si retorna error es qu'acabat i heu de tornar a excutar el codi d'a dalt
X_train, X_test, y_train, y_test, X_final_test,nom_dataset_selected = next(select_dataset)
print(f"Seleccionat dataset {nom_dataset_selected}")

# %%
results_df = pd.DataFrame(index=[], columns= ['Accuracy', 'F1 Macro', 'Precision Macro', 'Recall Macro'])

# %% [markdown]
# ## 1.LDA

# %%
# Train LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# %%
cross_val_results = pd.DataFrame(cross_validate(lda , X_train, y_train, cv = 5, 
                            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] )) #cv es el nom de folds de cross validation

results_df.loc['LDA',:] = cross_val_results[['test_accuracy', 'test_f1_macro',
       'test_precision_macro', 'test_recall_macro']].mean().values
results_df #scores de validacio

# %%
#LDA Separacio aplicada graficament
X_transformed = lda.transform(X_train)

X_transformed = pd.DataFrame(X_transformed)
X_transformed['labels'] = y_train.reset_index(drop=True)
X_transformed

# %%
sn.scatterplot(x= 0, y= 1, data = X_transformed, hue='labels',palette="coolwarm")

# %%
#TEST (No Final), executar despres de validacio de tots els models
y_test_lda_pred = lda.predict(X_test)

# %%
#Metriques
accuracy_lda = accuracy_score(y_test, y_test_lda_pred)
f1_lda = f1_score(y_test, y_test_lda_pred,average='macro')
print(f"LDA test accuracy: {accuracy_lda} \n LDA test f1-score: {f1_lda}")

# %%
cm_lda = confusion_matrix(y_test, y_test_lda_pred)
disp_lda = ConfusionMatrixDisplay(cm_lda)
fig, axs = plt.subplots(figsize=(12, 4))

disp_lda.plot(ax=axs)

axs.set_title('LDA')

# %% [markdown]
# ## 2.QDA

# %%
#Validacio d'hiperparametres
reg_param_values = [0, 0.0001, 0.001, 0.1, 1]

results_qda_df = pd.DataFrame(index=[], columns= ['reg_param', 'Accuracy', 'F1 Macro', 'Precision Macro', 'Recall Macro'])

for i, reg_value in enumerate(reg_param_values):
    qda = QuadraticDiscriminantAnalysis(reg_param=reg_value)
    try:
        cross_val_results = pd.DataFrame(cross_validate(qda , X_train, y_train, cv = 5, scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] ))
    except: #per si es singular
        continue
    else:
        results_qda_df.loc[f'{i}',:] = [reg_value] + list(cross_val_results[['test_accuracy', 'test_f1_macro',
                                                                             'test_precision_macro', 'test_recall_macro']].mean().values)

results_qda_df = results_qda_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
results_qda_df

# %%
best_reg_param = results_qda_df["reg_param"][0]
print(f"The best value for reg_param is {best_reg_param}.") #aquest valor es el bo, ull perque el que mostra el dataframe a vegades fa un round quan ho mostras

# %%
qda = QuadraticDiscriminantAnalysis(reg_param=best_reg_param)
qda.fit(X_train, y_train)
cm_qda = confusion_matrix(y_train, qda.predict(X_train))
disp_qda = ConfusionMatrixDisplay(cm_qda)
fig, axs = plt.subplots(figsize=(12, 4))

disp_qda.plot(ax=axs)

axs.set_title('QDA')

# %%
results_df.loc['QDA',:] = results_qda_df.loc[0, ['Accuracy', 'F1 Macro', 'Precision Macro', 'Recall Macro']]
results_df

# %%
#TEST (No Final), executar despres de validacio de tots els models
y_test_qda_pred = qda.predict(X_test)

# %%
accuracy_qda = accuracy_score(y_test, y_test_qda_pred)
f1_qda = f1_score(y_test, y_test_qda_pred,average='macro')
print(f"QDA test accuracy: {accuracy_qda} \n QDA test f1-score: {f1_qda}")

# %%
cm_qda = confusion_matrix(y_test, y_test_qda_pred)
disp_qda = ConfusionMatrixDisplay(cm_qda)
fig, axs = plt.subplots(figsize=(12, 4))

disp_qda.plot(ax=axs)

axs.set_title('QDA')

# %% [markdown]
# ## 3.Naive-Bayes Classifier

# %%
gaussian_nb = GaussianNB()

cross_val_results = pd.DataFrame(cross_validate(gaussian_nb , X_train, y_train, cv = 5, scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] ))

results_df.loc['Gaussian Naive Bayes',:] = cross_val_results[['test_accuracy', 'test_f1_macro',
       'test_precision_macro', 'test_recall_macro']].mean().values
results_df

# %%
gaussian_nb.fit(X_train, y_train)
cm_naive = confusion_matrix(y_train, pd.Series(gaussian_nb.predict(X_train)))
disp_naive = ConfusionMatrixDisplay(cm_naive)
fig, axs = plt.subplots(figsize=(12, 4))

disp_naive.plot(ax=axs)

axs.set_title('Naive-Bayes')

# %%
#TEST (No Final), executar despres de validacio de tots els models
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb,average='macro')

# %%
accuracy_nb



# %%
f1_nb

# %%
cm_naive = confusion_matrix(y_test, y_pred_nb)
disp_naive = ConfusionMatrixDisplay(cm_naive)
fig, axs = plt.subplots(figsize=(12, 4))

disp_naive.plot(ax=axs)

axs.set_title('Naive-Bayes')

# %% [markdown]
# ## 4.K-NN

# %%
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(
    estimator=knn,
    param_grid={
        'n_neighbors': [1, 3, 5, 7, 10, 15, 20],
        'metric': ['euclidean', 'minkowski', 'manhattan', 'cosine']
    },
    scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
    refit=False
)

knn_cv.fit(X_train, y_train)
results_knn_df = pd.DataFrame(knn_cv.cv_results_)
cols = ['param_n_neighbors', 'param_metric',
     'mean_test_accuracy',
    'mean_test_f1_macro', 'mean_test_precision_macro',
    'mean_test_recall_macro', 
    'std_test_accuracy', 'std_test_f1_macro', 'std_test_precision_macro',
    'std_test_recall_macro'
]
results_knn_df = results_knn_df[cols].sort_values(by='mean_test_accuracy',ascending=False).reset_index(drop=True)
results_knn_df

# %%
best_n_neighbors = results_knn_df["param_n_neighbors"][0]
best_metric = results_knn_df["param_metric"][0]
print(f'The best set of parameters is k={best_n_neighbors} and distance={best_metric}.')

# %%
knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, metric=best_metric)
knn_model.fit(X_train, y_train)

cm_knn = confusion_matrix(y_train, pd.Series(knn_model.predict(X_train)))
disp_knn = ConfusionMatrixDisplay(cm_naive)
fig, axs = plt.subplots(figsize=(12, 4))

disp_knn.plot(ax=axs)

axs.set_title('K-NN')

# %%
results_df.loc['KNN',:] = results_knn_df.loc[0, ['mean_test_accuracy', 'mean_test_f1_macro',
       'mean_test_precision_macro', 'mean_test_recall_macro']].values
results_df

# %%
#TEST (No Final), executar despres de validacio de tots els models
knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, metric=best_metric)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn,average='macro')

# %%
accuracy_knn

# %%
f1_knn

# %% [markdown]
# ## 5.Logistic Regression

# %%
logreg = LogisticRegressionCV(Cs=20, random_state=SEED, cv=5, scoring = 'accuracy', l1_ratios=[0],solver = 'lbfgs',use_legacy_attributes=False)

logreg.fit(X_train, y_train)

# %%
avg_crossval_scores = logreg.scores_.mean(axis=0)
idx = np.argmax(avg_crossval_scores)
best_C = logreg.Cs_[idx]
print(f'The best value for C is {best_C}.')

# %%
logreg = LogisticRegression(C=best_C, l1_ratio=0)
cross_val_results = pd.DataFrame(cross_validate(logreg, X_train, y_train, cv = 5, scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] ))

results_df.loc['Logistic Regression',:] = cross_val_results[['test_accuracy', 'test_f1_macro',
       'test_precision_macro', 'test_recall_macro']].mean().values

results_df.sort_values(by='Accuracy', ascending=False)

# %%
#TEST (No Final), executar despres de validacio de tots els models
logreg = LogisticRegression(C=best_C, l1_ratio=0)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg,average='macro')

# %%
accuracy_logreg

# %%
f1_logreg

# %% [markdown]
# ## 6.SVM

# %%
#Si C es massa gran pot-hi haver overfitting

# %%
#Setup de posibles valors dels hiperparametres
p_grid = {"C": [1, 10,50, 100], "gamma": [0.001,0.01, 0.1]}

# Utilitzem Support Vector Classifier amb "rbf" kernel
svm = SVC(kernel="rbf")


inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)


clf = GridSearchCV(estimator=svm, param_grid=p_grid,scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'], cv=inner_cv,refit=False)
clf.fit(X_train, y_train)
results_svc_df = pd.DataFrame(clf.cv_results_)

# %%
results_svc_df.columns

# %%
cols = ['param_C', 'param_gamma',
     'mean_test_accuracy',
    'mean_test_f1_macro', 'mean_test_precision_macro',
    'mean_test_recall_macro', 
    'std_test_accuracy', 'std_test_f1_macro', 'std_test_precision_macro',
    'std_test_recall_macro'
]
results_svc_df_proc = results_svc_df[cols].sort_values(by='mean_test_accuracy',ascending=False).reset_index(drop=True)
results_svc_df_proc

# %%
best_Cs = results_svc_df_proc["param_C"][0]
best_gamma = results_svc_df_proc["param_gamma"][0]
print(f'The best set of parameters is C={best_Cs} and gamma={best_gamma}.')

# %%
#Afegim resultats a la Taula
results_df.loc['SVC',:] = results_svc_df_proc.loc[0, ['mean_test_accuracy', 'mean_test_f1_macro',
       'mean_test_precision_macro', 'mean_test_recall_macro']].values
results_df

# %%
#TEST (No Final), executar despres de validacio de tots els models
svm = SVC(kernel='rbf', C=best_Cs, gamma=best_gamma, random_state=SEED)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average="macro")

# %%
accuracy_svm

# %%

f1_svm

# %%
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(cm_svm)
fig, axs = plt.subplots(figsize=(12, 4))

disp_svm.plot(ax=axs)

axs.set_title('SVM')

# %% [markdown]
# ## 7.Random forest

# %%
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=SEED,
    n_jobs=-1
)

param_grid = {
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

clf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
    cv=inner_cv,
    refit=False,
    n_jobs=-1
)

clf.fit(X_train, y_train)
results_rf_df = pd.DataFrame(clf.cv_results_)

# %%
results_rf_df.columns

# %%
cols = ['param_max_depth', 'param_max_features','param_min_samples_leaf','param_min_samples_split',
     'mean_test_accuracy',
    'mean_test_f1_macro', 'mean_test_precision_macro',
    'mean_test_recall_macro', 
    'std_test_accuracy', 'std_test_f1_macro', 'std_test_precision_macro',
    'std_test_recall_macro'
]
results_rf_df_proc = results_rf_df[cols].sort_values(by='mean_test_accuracy',ascending=False).reset_index(drop=True)
results_rf_df_proc

# %%
best_max_depth= results_rf_df_proc["param_max_depth"][0]
best_max_features = results_rf_df_proc["param_max_features"][0]
best_min_samples_leaf = results_rf_df_proc["param_min_samples_leaf"][0]
best_min_samples_split = results_rf_df_proc["param_min_samples_split"][0]
print(f'The best set of parameters are max_depth = {best_max_depth}, max_features = {best_max_features}, min_samples_leaf = {best_min_samples_leaf} and min_samples_split = {best_min_samples_split}.')

# %%
#Afegim resultats a la Taula
results_df.loc['RandomForest',:] = results_rf_df_proc.loc[0, ['mean_test_accuracy', 'mean_test_f1_macro',
       'mean_test_precision_macro', 'mean_test_recall_macro']].values
results_df

# %%
#TEST (No Final), executar despres de validacio de tots els models
rf_model = RandomForestClassifier(n_estimators=200, max_depth=best_max_depth, max_features= best_max_features, min_samples_leaf=best_min_samples_leaf,
                                  min_samples_split=best_min_samples_split, random_state=SEED, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')

# %%
acc_rf



# %%
f1_rf

# %% [markdown]
# ## 8. Gradient boosting
# Pel gradient boosting com es un classificador que triga molt més qu'els altres, ho farem amb cross validate i els hiperparametres ja els hem fixat a ma

# %%
gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=SEED)

cross_val_results = pd.DataFrame(cross_validate(gb_model, X_train, y_train, cv = 5, scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] ))

results_df.loc['GradientBoosting',:] = cross_val_results[['test_accuracy', 'test_f1_macro',
       'test_precision_macro', 'test_recall_macro']].mean().values

# %%
results_df

# %%
#TEST (No Final), executar despres de validacio de tots els models
gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=SEED)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='macro')

# %%
acc_gb

# %%
f1_gb

# %% [markdown]
# ## Emmagatzematge dels millors models per a cada dataset

# %%
#Guardem la imatge dels resultats del dataframe
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
table = ax.table(cellText=results_df.values,
                 colLabels=results_df.columns,
                 rowLabels=results_df.index,
                 loc='center')
plt.title(f"Resultats models amb dataset {nom_dataset_selected}", fontsize=14, fontweight='bold', pad=20)
plt.savefig(f"results_df_{nom_dataset_selected}.png", bbox_inches='tight', dpi=300)

# %%
results_df.sort_values(by='F1 Macro',ascending=False) #ordenem per F1 Macro i ho emagatzemem a

# %%
#Emagatzemem el resultat al dataframe de millors resultats per dataframe
best_result = results_df.sort_values(by='F1 Macro',ascending=False).iloc[0]
best_datasets_result_df.loc[nom_dataset_selected,:] = np.concatenate(([best_result.name], best_result.values))
best_datasets_result_df

# %% [markdown]
# Ara cal tornar a executar tots els metodes per el seguents datasets fins arribar al final. Un cop arribats:

# %%
#Guardem la imatge dels resultats del dataframe
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
table = ax.table(cellText=best_datasets_result_df.values,
                 colLabels=best_datasets_result_df.columns,
                 rowLabels=best_datasets_result_df.index,
                 loc='center')
plt.title(f"Millors resultats models per a cada dataset", fontsize=14, fontweight='bold', pad=20)
plt.savefig(f"best_results_df.png", bbox_inches='tight', dpi=300)

# %% [markdown]
# ## 10. Execució sobre Final Test i Exportar resultats
# Ens quedem amb el millor model i dataset

# %%
best_datasets_result_df.sort_values(by='F1 Macro',ascending=False).iloc[0]

# %%
#Seleccionem el millor model
X_train, X_test, y_train, y_test, X_final_test,nom_dataset_selected = X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered, final_test_filtered(), "csi_filtered"

# %%
#Resultat sobre la Particio Test de les dades de train
best_C = 100
best_gamma = 0.1
svm = SVC(kernel='rbf', C=best_Cs, gamma=best_gamma, random_state=SEED)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average="macro")

print(f"Obtenim una Accuracy de {accuracy_svm} i un F1-score de {f1_svm}")

# %%
#Entrenem sobre tot el dataset de Train
X_full_train = pd.concat([X_train,X_test])
y_full_train = pd.concat([y_train,y_test])

best_C = 100
best_gamma = 0.1
svm = SVC(kernel='rbf', C=best_Cs, gamma=best_gamma, random_state=SEED)
svm.fit(X_full_train, y_full_train)

# %%
#Prediccio final pel kaggle
y_final_pred = svm.predict(X_final_test)
y_final_pred


# %%
output_submission(y_final_pred,"final_prediction")


