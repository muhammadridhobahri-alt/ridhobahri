import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("processed_kelulusan.csv")
x = df.drop("Lulus", axis=1)
y = df["Lulus"]

x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3,stratify=y, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42)

print(x_train.shape, x_val.shape, x_test.shape)

from sklearn.pipeline import Pipeline    
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

num_cols = x_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
], remainder="drop")

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])


pipe_lr.fit(x_train, y_train)
y_val_pred = pipe_lr.predict(x_val)
print("Baseline (logreg) f1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)

pipe_rf = Pipeline([("pre", pre),("clf", rf)])
pipe_rf.fit(x_train, y_train)
y_val_rf = pipe_rf.predict(x_val)
print("RandomForest f1(val):", f1_score(y_val, y_val_rf, average="macro"))

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

skf = StratifiedKFold(
    n_splits = 2,
    shuffle = True,
    random_state = 42
)

param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(
    pipe_rf,
    param_grid = param,
    cv = skf,
    scoring = "f1_macro",
    n_jobs = -1,
    verbose = 1
)

gs.fit(
    x_train,
    y_train
)
print("Best params:", gs.best_params_)
print("Best cv f1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(x_val)
print("Best rf f1(val):", f1_score(y_val, y_val_best, average="macro"))

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

final_model = best_rf
y_test_pred = final_model.predict(x_test)
print("f1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("confussion matrix(test):")
print(confusion_matrix(y_test, y_test_pred, labels=[0,1]))

if hasattr(final_model, "predict_proba"):
    if len(set(y_test)) == 2:
        y_test_proba = final_model.predict_proba(x_test)[:, 1]
        try :
            print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
        except :
            pass
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC(test)")
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi = 120)
    

import joblib
joblib.dump(final_model, "model.pkl")
print("Model tersimpan ke model.pkl")

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
MODEL = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    x = pd.DataFrame([data])
    yhat = MODEL.predict(x)[0]
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(x)[:, 1][0])
    return jsonify({"prediction": int(yhat), "proba": proba})

if __name__ == "__main__":
    app.run(port=5000)