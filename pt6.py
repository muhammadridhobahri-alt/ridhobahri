import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Baca data
df = pd.read_csv("processed_kelulusan.csv")
x = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split 70/15/15
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.30, stratify=y, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.50, random_state=42
)
print(x_train.shape, x_val.shape, x_test.shape)

# Preprocessing
num_cols = x_train.select_dtypes(include="number").columns
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols)
], remainder="drop")

# Model
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe = Pipeline([
    ("pre", pre),
    ("clf", rf)
])

pipe.fit(x_train, y_train)

# Evaluasi baseline
y_val_pred = pipe.predict(x_val)
print("Baseline RF - F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# Cross-validation
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scores = cross_val_score(pipe, x_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro(train):", scores.mean(), "±", scores.std())

# Grid Search
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(pipe, param_grid=param, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(x_train, y_train)

print("Best params:", gs.best_params_)
best_model = gs.best_estimator_

# Validasi
y_val_best = best_model.predict(x_val)
print("Best RF - F1(val):", f1_score(y_val, y_val_best, average="macro"))

# Tes akhir
final_model = best_model
y_test_pred = final_model.predict(x_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix(test):")
print(confusion_matrix(y_test, y_test_pred, labels=[0, 1]))

# ROC-AUC (jika 2 kelas)
if hasattr(final_model, "predict_proba"):
    if len(set(y_test)) == 2:
        y_test_proba = final_model.predict_proba(x_test)[:, 1]
        try:
            print("ROC-AUC(test):", roc_auc_score(y_test, y_test_proba))
        except:
            print("Terjadi error saat menghitung ROC-AUC.")
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC(test)")
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi=120)
    else:
        print("ROC-AUC tidak bisa dihitung karena hanya ada satu kelas di y_test.")

# Feature importance
try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    top = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    print("Top feature importance:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# Simpan model
joblib.dump(final_model, "rf_model.pkl")
print("Model disimpan sebagai rf_model.pkl")

# Coba prediksi
mdl = joblib.load("rf_model.pkl")
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4*7
}])
print("Prediksi:", int(mdl.predict(sample)[0]))