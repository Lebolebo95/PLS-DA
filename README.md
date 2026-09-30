# PLS-DA

Semplice classificatore PLS-DA compatibile con scikit-learn, costruito su
`PLSRegression`.

Supporta i metodi di classificazione `argmax`, `centroids`, `mahalanobis`,
`lda` e `qda` e permette di recuperare la varianza spiegata di X e Y.

## Utilizzo su Google Colab

```python
!wget -q https://raw.githubusercontent.com/Lebolebo95/PLS-DA/main/plsda.py

from plsda import PLSDA
```

## Esempio

```python
model = PLSDA(
    n_components=2,
    classification_method="argmax",
    scale=True,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.explained_variance_x())
print(model.explained_variance_y())
```

Le varianze spiegate sono restituite come quota incrementale, tra 0 e 1, per
ciascuna componente PLS.

## Requisiti

- NumPy
- scikit-learn
