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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# Set the precision of the display to 3 decimal places
pd.set_option('display.precision', 3)

# %%
#Constants
SEED:int = 383006
N_COMP:float = 0.99

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
# podriem treure rssi1 o rssi2 ja que correlan linealment?

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
csi[csi.rssi1 < -95] #eliminar observacio? cal tenir en compte que esta bastant a prop del -95

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
csi_filtered.describe()

# %% [markdown]
# # Els següents dataset tretes les I's i Q's 0:
# - csi_filtered: Cap canvi adicional
# - csi_uncorr: Es converteixen a Modul i Angle + Aplicacio de uncorr_vars()
# - csi_filter_plus: S'aplica directament min_mutual_info()
# - csi_pca: Aplicat PCA
# - csi_angle_pca: Modul i Angle + PCA

# %% [markdown]
# Ara estudiem les que no ho son.
# 
# Segons la informació que hem trobat sobre aquestes dades d'[OFDM]("https://www.cwnp.com/understanding-ofdm-part-2-2/"):
# 
# Es tracta de nombres complexos de la forma $H= I + iQ$, on $i^2=-1$.
# 
# Anem sustituir les variables I, Q per A (módul) i Theta (angle):
# 

# %%
def complex_conversion(df:pd.DataFrame):
    """Transforma les dade I i Q a A i O.

    Avis: Modifica el dataset donat com a parametre.

    Prec: Cal previament eliminar les I's i Q's que siguin 0"""
    for k in (1,2):
        j_iter:list[int] = [i for i in range(1,27)] + [i for i in range(38,64)]
        for j in j_iter:
            mod:str = f"A{j}_{k}"
            angl:str = f"O{j}_{k}"
            i_str = f"I{j}_{k}"
            q_str = f"Q{j}_{k}"
            complex = df[i_str] + df[q_str] * 1j
            np.angle(complex)
            df[mod] = np.abs(complex)
            df[angl] = np.angle(complex)
            df.drop(columns=[i_str,q_str],inplace=True)

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
#Algorisme com Sieve d' Eratosthenes pero per descartar les variables
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
j_iter:list[int] = [i for i in range(1,27)] + [i for i in range(38,64)] #per si cal en altres apartats aquest indexs

# %%
#csi_uncorr
complex_conversion(csi_uncorr)
moduls_1 = ["position"]+ [f"A{j}_1" for j in j_iter]
uncorr_moduls_1 = uncorr_vars(csi_uncorr,moduls_1[1:]) #de 52 variables incialment ens quedem amb 5
moduls_2 = [f"A{j}_2" for j in j_iter]
uncorr_moduls_2 = uncorr_vars(csi_uncorr,moduls_2) # de 52 variables incialment ens quedem amb 12
uncorr_moduls = uncorr_vars(csi_uncorr,uncorr_moduls_1 + uncorr_moduls_2)
angl_1 = [f"O{j}_1" for j in j_iter]
angl_2 = [f"O{j}_2" for j in j_iter]
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
sn.pairplot(data=csi_filtered[moduls_1[:len(moduls_1)//2]], hue='position',palette="coolwarm",corner=True)

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
def scaling_preprocessing(X, scaler=None)->tuple[pd.DataFrame,MinMaxScaler]: #funcio extraida de la practica 4 Linear Regression
    """Escala los datos numericos de X y los devuelve escalados y con su escalador.
    
    Prec: X no debe tener variables categoricas ni NA's

    :param: scaler: se debe indicar el utilizado en los datos de train cuando se utilicen los de test
    """
    print('Original shape:{}'.format(X.shape))
    categorical_columns = X.dtypes[X.dtypes == 'category'].index.values
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
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=seed, stratify=csi_filtered['position']) #stratify fara que es conservin les mateixes proporcions de la feature position
    return (X_train, X_test, y_train, y_test)

# %%
#csi_filtered
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = df_train_test_split(csi_filtered)
X_train_filtered, scaler_filtered = scaling_preprocessing(X_train_filtered)
X_test_filtered, _ = scaling_preprocessing(X_test_filtered,scaler_filtered)

# %%
#csi_uncorr
X_train_uncorr, X_test_uncorr, y_train_uncorr, y_test_uncorr = df_train_test_split(csi_uncorr)
X_train_uncorr, scaler_uncorr = scaling_preprocessing(X_train_uncorr)
X_test_uncorr, _ = scaling_preprocessing(X_test_uncorr,scaler_uncorr)

# %%
#csi_filter_plus
X_train_filter_plus, X_test_filter_plus, y_train_filter_plus, y_test_filter_plus = df_train_test_split(csi_filter_plus)
X_train_filter_plus, scaler_filter_plus = scaling_preprocessing(X_train_filter_plus)
X_test_filter_plus, _ = scaling_preprocessing(X_test_filter_plus,scaler_filter_plus)

# %%
#csi_pca
X_train_pca, X_test_pca, y_train_pca, y_test_pca = df_train_test_split(csi_pca)
X_train_pca.shape

# %%
#csi_angle_pca
X_train_angle_pca, X_test_angle_pca, y_train_angle_pca, y_test_angle_pca = df_train_test_split(csi_angle_pca)
X_train_angle_pca.shape

# %%


# %% [markdown]
# ### Balancejar el dataset
# Com que hem considerat qu'el dataset estaba balancejat perquè no hi havien grans diferencies per les classes, amb això acabem el Dataset Cleaning, ara comencarem a entrenar models.

# %%


# %% [markdown]
# ### Funcions per l'exportació a Kaggle

# %%
#funcions per transformar csi_final_test per que sigui com X i y (trasnformacions i reduccions de variables inclosas)
def final_test_filtered(df_raw:pd.DataFrame=csi_final_test,scaler:MinMaxScaler=scaler_filtered,columns_to_drop:list[str]=carriers_0)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de filtered.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop,inplace=True) # fem un drop de les I's i Q's igual a 0
    X_final_test, _ = scaling_preprocessing(df,scaler)
    return X_final_test

# %%
def final_test_uncorr(df_raw:pd.DataFrame=csi_final_test,scaler:MinMaxScaler=scaler_uncorr,columns_to_drop:list[str]=carriers_0 + vars_to_drop_uncorr)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de uncorr.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop[:48],inplace=True) # fem un drop de les I's i Q's igual a 0
    complex_conversion(df)
    df.drop(columns=columns_to_drop[48:],inplace=True) #treim les variable no correlades
    X_final_test, _ = scaling_preprocessing(df,scaler)
    return X_final_test

# %%
def final_test_filter_plus(df_raw:pd.DataFrame=csi_final_test,scaler:MinMaxScaler=scaler_filter_plus,columns_to_drop:list[str]=carriers_0 + vars_to_drop_filter_plus)->pd.DataFrame:
    """Retorna un Dataframe apte per predir segons les transformacions de filter_plus.
    
    Prec: No cal incloure 'ID' a columns_to_drop, ja ho fa automaticament"""
    df = df_raw.drop(columns="ID")
    df.drop(columns=columns_to_drop,inplace=True)
    X_final_test, _ = scaling_preprocessing(df,scaler)
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
#funcions per escriure el resultat a un fitxer
def output_submission(y:np.ndarray,filename:str="out")->None:
    """Escribe y en el fichero filename.csv para la submission. No hay que incluir '.csv' en filename"""
    filename= filename + ".csv"
    with open(filename,"w") as f:
        print("ID,POSITION",file=f)
        for i in range(len(y)):
            print(f"{i},{y[i]}",file=f)
    print("Fitxer d'output generat")

# %%
prueba_arr = carriers_0 + vars_to_drop_uncorr

# %%
vars_to_drop_uncorr == prueba_arr[48:]

# %%


# %% [markdown]
# # Seleccio de Dades per l'entrenament

# %%
#Exemple (es pot canviar per X_train_..., etc)
X_train = X_train_uncorr
X_test = X_test_uncorr
y_train = y_train_uncorr
y_test = y_test_uncorr

# %% [markdown]
# ## 1.LDA

# %%
# Train LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# %%
#Afegir validacio

# %%
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

# %%
lda.fit(X_train_pca, y_train_pca)
y_test_lda_pred = lda.predict(X_test_pca)
accuracy_lda = accuracy_score(y_test_pca, y_test_lda_pred)
f1_lda = f1_score(y_test, y_test_lda_pred,average='macro')
print(f"LDA test accuracy: {accuracy_lda} \n LDA test f1-score: {f1_lda}")

# %%


# %% [markdown]
# ## 2.QDA

# %%
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train,y_train)

# %%
y_test_qda_pred = qda.predict(X_test)

# %%
accuracy_qda = accuracy_score(y_test, y_test_qda_pred)
f1_qda = f1_score(y_test, y_test_qda_pred,average='macro')
print(f"QDA test accuracy: {accuracy_qda} \n QDA test f1-score: {f1_qda}")

# %%
cm_qda = confusion_matrix(y_test, y_test_lda_pred)
disp_qda = ConfusionMatrixDisplay(cm_qda)
fig, axs = plt.subplots(figsize=(12, 4))

disp_qda.plot(ax=axs)

axs.set_title('QDA')

# %%
qda.fit(X_train_pca, y_train)
y_test_qda_pred = lda.predict(X_test_pca)
accuracy_qda = accuracy_score(y_test_pca, y_test_qda_pred)
f1_qda = f1_score(y_test, y_test_qda_pred,average='macro')
print(f"LDA test accuracy: {accuracy_qda} \n LDA test f1-score: {f1_qda}")

# %%


# %%


# %% [markdown]
# ## 3.Naive-Bayes Classifier

# %%
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_test_qda_pred,average='macro')

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

# %%


# %% [markdown]
# ## 4.Neural Probabilistic Classifier

# %%


# %% [markdown]
# ## 5.K-NN

# %%


# %% [markdown]
# ## 6.Logistic Regression

# %%


# %% [markdown]
# ## 7.SVM

# %%


# %%
X_train = X_train_filtered
X_test = X_test_filtered
y_train = y_train_filtered
y_test = y_test_filtered

# %%
svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=SEED)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average="macro")

# %%
accuracy_svm

# %%

f1_svm

# %%
X_final_test = final_test_filtered()
X_final_test

# %%
y_final_pred = svm.predict(X_final_test)


# %%
y_final_pred.shape

# %%


# %% [markdown]
# ## 8.Random forest and Gradient boosting

# %%
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')

# %%
acc_rf



# %%
f1_rf

# %%
rf_model_pca = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
rf_model_pca.fit(X_train_pca, y_train)

y_pred_rf_pca = rf_model_pca.predict(X_test_pca)
acc_rf_pca = accuracy_score(y_test, y_pred_rf_pca)
f1_rf_pca = f1_score(y_test, y_pred_rf_pca, average='macro')

# %%
acc_rf_pca

# %%
f1_rf_pca

# %%
gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='macro')

# %%
acc_gb

# %%
f1_gb

# %%
gb_model_pca = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
gb_model_pca.fit(X_train_pca, y_train)

y_pred_gb_pca = gb_model_pca.predict(X_test_pca)
acc_gb_pca = accuracy_score(y_test, y_pred_gb_pca)
f1_gb_pca = f1_score(y_test, y_pred_gb_pca, average='macro')

# %%
acc_gb_pca

# %%
f1_gb_pca

# %% [markdown]
# ## 8.Mirar més métodes a la resta de transpas

# %%


# %% [markdown]
# # Temporal Cache de Dades (eliminar al final)
# A continuacio hi han alguns plots que vam fer durant l'AED però qu'al final no vam utilitzar 

# %%
sn.pairplot(data=csi[["seq_ctrl","aoa","rssi1","rssi2","position", "I1_1", "Q1_1", "I63_1", "Q63_1", "I1_2", "Q1_2", "I63_2", "Q63_2"]], hue='position',palette="coolwarm")

# %% [markdown]
# Observem algunes relacions lineals entre variables I_x_n, anem a fer un correlation plot de totes les variables I_x_1 $\neq 0$

# %%
I_1 = ["position"] + [f"I{i}_1" for i in range(1,27)] + [f"I{i}_1" for i in range(38,64)]

# %%
sn.pairplot(data=csi[I_1])

# %% [markdown]
# Observem que per les I_x_1 que son aprop (diferencia de la x no molt gran), la relació sembla ser lineal, però quan es comença a allunyar aquesta relació lineal es perd.

# %%
#podria ser que I_x fossi una time series d'algo del wifi?


