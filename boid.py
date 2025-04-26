## Simulation interraction d'oiseaux via le modèle de Boids

# Importation des bibliothèques nécessaires
import taichi as ti

# Initialisation de Taichi avec l'architecture GPU
ti.init(arch=ti.gpu)

# Définition des paramètres de simulation
Hauteur, Largeur = 1000, 2000
Taille_memoire = 200
Nbr_oiseaux = 1000

# Définition des champs de positions et de vitesses et de couleur des oiseaux
position = ti.Vector.field(2, dtype=ti.f32, shape=Nbr_oiseaux)
vitesse = ti.Vector.field(2, dtype=ti.f32, shape=Nbr_oiseaux)
couleur = ti.Vector.field(3, dtype=ti.f32, shape=Nbr_oiseaux)

# Définition des champs de mémoire pour les oiseaux
position_memoire = ti.Vector.field(2, dtype=ti.f32, shape=(Taille_memoire, Nbr_oiseaux))
vitesse_memoire = ti.Vector.field(2, dtype=ti.f32, shape=(Taille_memoire, Nbr_oiseaux))

# Paramètres de simulation variables (curseurs)


