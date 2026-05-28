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
# i Qn_X: Valor Quadraturedel subcarrier n-èssim a la antena X
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
#Histograma i Boxplot de rssi2
fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios':[1,4]}, figsize=(9,5))
csi.boxplot(column = "rssi2", ax=axes[0])
csi.hist(column = "rssi2", ax = axes[1])

# %%
#Histograma i Boxplot de AOA
fig, axes= plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(9,5))
csi.boxplot(column='aoa',ax=axes[0])
csi.hist(column='aoa', ax=axes[1])

# %% [markdown]
# Semblaría que no hi han outliers per aquestes variables

# %% [markdown]
# Estudiem les dades RAW CSI:

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
# Ara estudiem les que no ho son:

# %%
sn.pairplot(data=csi[["seq_ctrl","aoa","rssi1","rssi2","position", "I1_1", "Q1_1", "I63_1", "Q63_1", "I1_2", "Q1_2", "I63_2", "Q63_2"]], hue='position',palette="coolwarm")

# %% [markdown]
# Observem algunes relacions lineals entre variables I_x_n, anem a fer un correlation plot de totes les variables I_x_1 $\neq 0$

# %%
I_1 = ["position"] + [f"I{i}_1" for i in range(1,27)] + [f"I{i}_1" for i in range(38,64)]
I_1

# %%
sn.pairplot(data=csi[I_1])

# %% [markdown]
# Observem que per les I_x_1 que son aprop (diferencia de la x no molt gran), la relació sembla ser lineal, però quan es comença a allunyar aquesta relació lineal es perd.

# %%
#podria ser que I_x fossi una time series d'algo del wifi?

# %%


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



