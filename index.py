import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv('salarios.csv')

df = df.drop(['matricula', 'lotacao', 'detalhamento_contracheque', 'nome', 'cargo'], axis=1)

df = df[(df['valor'] > 0)]

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.3),max_features=1.0)
model.fit(df[['valor']])

df['scores']=model.decision_function(df[['valor']])
df['anomaly']=model.predict(df[['valor']])

anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)

anomaly.to_csv('outliers.csv')
