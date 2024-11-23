import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
from matplotlib.animation import FuncAnimation


data = loadmat('Test2.mat')
t, x1, x2 = data['t'], data['x1'], data['x2']

custom_cmap = ListedColormap(['#A2D2DF', '#F6EFBD'])

# Matrice augmentée
X = np.column_stack((x1, x2, np.ones((t.shape))))  # Ajouter le biais (colonne de 1)
N, d = X.shape

# Initialisation
w = np.ones(d)  # Poids initialisés à zéro
delta = 1  # Pas d'apprentissage
epochs = 50  # Nombre total d'epochs

# Préparer les données pour l'animation
history = []  # Stocke les poids w à chaque étape

for epoch in range(epochs):
    mal_classe = 0
    for i in range(N):
        if np.sign(t[i] * w.T @ X[i, :]) < 0:
            mal_classe += 1
            w = w + delta * t[i] * X[i, :].T  # Mise à jour des poids
    history.append(w.copy())  # Sauvegarder les poids pour l'animation

# Préparer la figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1[t.flatten() == 1], x2[t.flatten() == 1], color="#BFECFF", label="Classe 1")
ax.scatter(x1[t.flatten() == -1], x2[t.flatten() == -1], color="#FFCCEA", label="Classe -1")
decision_line, = ax.plot([], [], 'k-', lw=2, label="Hyperplan")
ax.set_xlim(x1.min() - 1, x1.max() + 1)
ax.set_ylim(x2.min() - 1, x2.max() + 1)
ax.set_title("Évolution de l'hyperplan au fil des epochs")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()
ax.grid(alpha=0.3)

# Fonction pour mettre à jour l'animation
def update(frame):
    w = history[frame]
    x1_range = np.linspace(x1.min(), x1.max(), 100)
    if w[1] != 0:  # Pour éviter la division par zéro
        x2_range = -(w[0] * x1_range + w[2]) / w[1]
        decision_line.set_data(x1_range, x2_range)
    ax.set_title(f"Évolution de l'hyperplan - Epoch {frame + 1}")
    return decision_line,

# Animation
ani = FuncAnimation(fig, update, frames=len(history), interval=1000, blit=True)

# Afficher l'animation dans le notebook
plt.show()
