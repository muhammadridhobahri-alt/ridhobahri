import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("processed_kelulusan.csv")
x =  df.drop("Lulus", axis=1)
y =  df["Lulus"]

sc = StandardScaler()
xs = sc.fit_transform(x)

x_train, x_temp, y_train, y_temp = train_test_split(
    xs, y, test_size=0.3,stratify=y, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape )

import tensorflow as tf
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense (32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid") #klasifikasi biner
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy","AUC"])
model.summary()

es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[es], verbose=1
)
from sklearn.metrics import classification_report, confusion_matrix

loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
print("Test acc:", acc, "AUC", auc)

y_proba = model.predict(x_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print (confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))

import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Learning Curve")
plt.tight_layout(); plt.savefig("learning_curve.png", dpi=120)