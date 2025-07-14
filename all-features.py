import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

MODEL_PATH = "model.keras"
LR_FILE = "lr.txt"

dtype = np.float64

def read_learning_rate(default=0.0001):
    if os.path.exists(LR_FILE):
        with open(LR_FILE, "r") as f:
            return float(f.read().strip())
    return default

def save_learning_rate(lr):
    with open(LR_FILE, "w") as f:
        f.write(str(lr))

def build_model(learning_rate):
    model = Sequential([
        Input(shape=(1,)),
        Dense(256, activation='relu'),
        #Dense(256, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Callback personnalisé pour tracer LR et loss à chaque epoch
class LrLossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.lrs = []
        self.losses = []
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        loss = logs.get('loss')
        self.lrs.append(lr)
        self.losses.append(loss)
        save_learning_rate(lr)  # Sauvegarde aussi learning rate
    def plot(self):
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Learning Rate', color=color)
        ax2.plot(self.lrs, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Learning Rate et Loss par Epoch")
        plt.show()

def train_model(model, entree, sortie, epochs=10000):
    early_stop = EarlyStopping(monitor='loss', patience=300, min_delta=1e-9, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, verbose=1)
    history = LrLossHistory()

    model.fit(entree, sortie, epochs=epochs, verbose=1, batch_size=256, callbacks=[early_stop, reduce_lr, history])
    model.save(MODEL_PATH)
    print(f"\n✅ Modèle sauvegardé dans {MODEL_PATH}")

    history.plot()

def main():
    entree = np.array(range(1, 10000), dtype=dtype)
    sortie = 2 * entree

    if os.path.exists(MODEL_PATH):
        print(f"📁 Modèle trouvé ({MODEL_PATH})")
        choix = input("Tape 't' pour tester, 'e' pour Fine-Tuning (reprise avec LR réduit) : ").strip().lower()
        model = load_model(MODEL_PATH)

        if choix == 't':
            print("🔍 Mode test activé.")
        elif choix == 'e':
            old_lr = read_learning_rate()
            new_lr = old_lr * 0.5
            print(f"📉 Learning rate précédent : {old_lr:.2e} → Nouveau : {new_lr:.2e}")

            optimizer = Adam(learning_rate=new_lr)
            model.compile(loss='mean_squared_error', optimizer=optimizer)

            train_model(model, entree, sortie, epochs=3000)
            save_learning_rate(new_lr)
        else:
            print("❌ Choix invalide. Arrêt.")
            return
    else:
        print("🛠️ Pas de modèle trouvé. Création initiale.")
        initial_lr = 0.0001
        model = build_model(initial_lr)
        train_model(model, entree, sortie, epochs=1000)
        save_learning_rate(initial_lr)

    try:
        while True:
            x = float(input("Nombre >>> "))
            prediction = model.predict(np.array([[x]], dtype=dtype), verbose=0)
            print("Prédiction :", prediction[0][0])
    except KeyboardInterrupt:
        print("\n👋 Programme interrompu.")

if __name__ == "__main__":
    main()
