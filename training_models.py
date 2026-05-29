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
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# Set the precision of the display to 3 decimal places
pd.set_option('display.precision', 3)

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
csi_filtered.describe()

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
for k in (1,2):
    j_iter:list[int] = [i for i in range(1,27)] + [i for i in range(38,64)]
    for j in j_iter:
        mod:str = f"A{j}_{k}"
        angl:str = f"O{j}_{k}"
        i_str = f"I{j}_{k}"
        q_str = f"Q{j}_{k}"
        complex = csi_filtered[i_str] + csi_filtered[q_str] * 1j
        np.angle(complex)
        
        csi_filtered[mod] = np.abs(complex)
        csi_filtered[angl] = np.angle(complex)
        csi_filtered.drop(columns=[i_str,q_str],inplace=True)

# %%
csi_filtered.describe()

# %%
moduls_1 = ["position"]+ [f"A{j}_1" for j in j_iter]
moduls_1[:len(moduls_1)//2]

# %%
sn.pairplot(data=csi_filtered[moduls_1[:len(moduls_1)//2]], hue='position',palette="coolwarm",corner=True)

# %%
len(moduls_1)

# %%


# %%
#Algorisme com Sieve d' Eratosthenes pero per descartar les variables
def uncorr_vars(df:pd.DataFrame,vars:list[str],min_corr:float=0.6, drop_inplace:bool=False)->list[str]: #O(n_vars**2 * columns(df) * rows(df))
    """Devuelve las variables que poseen una abs(correlacion) < min_corr.
    
    :param: drop_inplace: si es True, entonces elimina las variables correlacionadas
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
    if drop_inplace:
        df.drop(columns=[vars[j] for j in range(n_vars) if not(vars_select[j])])
    return [vars[j] for j in range(n_vars) if vars_select[j]]


# %%
uncorr_moduls_1 = uncorr_vars(csi_filtered,moduls_1[1:]) #de 52 variables incialment ens quedem amb 5
uncorr_moduls_1

# %%
moduls_2 = [f"A{j}_2" for j in j_iter]
uncorr_moduls_2 = uncorr_vars(csi_filtered,moduls_2) # de 52 variables incialment ens quedem amb 12
uncorr_moduls_2

# %%
uncorr_moduls = uncorr_vars(csi_filtered,uncorr_moduls_1 + uncorr_moduls_2)
uncorr_moduls #pasem de 104 variables originalment a 17

# %%
len(uncorr_moduls)

# %%
angl_1 = [f"O{j}_1" for j in j_iter]
angl_2 = [f"O{j}_2" for j in j_iter]

# %%
uncorr_angl_1 = uncorr_vars(csi_filtered,angl_1) # de 52 vars pasem a 2
uncorr_angl_1

# %%
uncorr_angl_2 = uncorr_vars(csi_filtered,angl_2) # de 52 vars pasem a 1
uncorr_angl_2

# %%
uncorr_raw_csi = uncorr_moduls + uncorr_angl_1 + uncorr_angl_2
len(uncorr_raw_csi)

# %%
len(uncorr_vars(csi_filtered,uncorr_raw_csi)) #comprobacio final que aquestes variables no estan correlades

# %% [markdown]
# Hem aconseguit reduir les 256 variables Raw CSI incialment presents per 20 variables.

# %% [markdown]
# ### Escalar les dades

# %%


# %%


# %%


# %%


# %%


# %%


# %%
csi_train, csi_test = train_test_split(csi, test_size=0.25, random_state=42, stratify=csi['position']) #stratify fara que es conservin les mateixes proporcions de la feature position

# %% [markdown]
# ## 1.LDA

# %%


# %% [markdown]
# ## 2.QDA

# %%


# %% [markdown]
# ## 3.Naive-Bayes Classifier

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


