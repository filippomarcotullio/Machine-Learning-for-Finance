#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance scikit-learn pandas numpy matplotlib')


# In[3]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
plt.style.use("default")
TICKER = "AAPL"          
START = "2015-01-01"
END = "2024-12-31"


# In[4]:


data = yf.download(TICKER, start=START, end=END)

data = data[["Close"]].dropna()
data.tail()


# In[5]:


data["ret"] = data["Close"].pct_change()

data["target"] = (data["ret"].shift(-1) > 0).astype(int)

data["ret_1"] = data["ret"].shift(1)
data["ret_2"] = data["ret"].shift(2)
data["ret_5_mean"] = data["ret"].shift(1).rolling(5).mean()
data["ret_5_vol"] = data["ret"].shift(1).rolling(5).std()

data = data.dropna()

data.head()


# In[6]:


feature_cols = ["ret_1", "ret_2", "ret_5_mean", "ret_5_vol"]

X = data[feature_cols].values
y = data["target"].values

len(X), len(y)


# In[7]:


split_idx = int(len(X) * 0.7)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

len(X_train), len(X_test)


# In[8]:


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
print("Test  accuracy:", round(accuracy_score(y_test,  y_test_pred), 4))

print("\nClassification report (test set):")
print(classification_report(y_test, y_test_pred))


# In[10]:


data_test = data.iloc[split_idx:].copy()
data_test["pred"] = y_test_pred

data_test["position"] = data_test["pred"]

data_test["strategy_ret"] = data_test["position"] * data_test["ret"]

data_test["bh_equity"] = (1 + data_test["ret"]).cumprod()   # buy & hold
data_test["ml_equity"] = (1 + data_test["strategy_ret"]).cumprod()

data_test[["bh_equity", "ml_equity"]].head()


# In[11]:


plt.figure(figsize=(10, 5))
plt.plot(data_test.index, data_test["bh_equity"], label="Buy & Hold")
plt.plot(data_test.index, data_test["ml_equity"], label="ML Strategy")

plt.title(f"{TICKER} - Buy & Hold vs ML Strategy")
plt.xlabel("Date")
plt.ylabel("Equity (starting at 1.0)")
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


bh_final = data_test["bh_equity"].iloc[-1]
ml_final = data_test["ml_equity"].iloc[-1]

print("Final equity Buy & Hold:", round(bh_final, 3))
print("Final equity ML Strategy:", round(ml_final, 3))

