## Simulation interraction d'oiseaux via le modèle de Boids

# Importation des bibliothèques nécessaires
import taichi as ti
import taichi.math as mti
from math import pi 

# Initialisation de Taichi avec l'architecture GPU
ti.init(arch=ti.gpu)

# Définition des paramètres de simulation
Hauteur, Largeur = 1000, 2000
Taille_memoire = 200

# Définition des champs de positions et de vitesses et de couleur des oiseaux
position = ti.Vector.field(2, dtype=ti.f32, shape=Nbr_oiseaux)
vitesse = ti.Vector.field(2, dtype=ti.f32, shape=Nbr_oiseaux)
couleur = ti.Vector.field(3, dtype=ti.f32, shape=Nbr_oiseaux)

# Définition des champs de mémoire pour les oiseaux
position_memoire = ti.Vector.field(2, dtype=ti.f32, shape=(Taille_memoire, Nbr_oiseaux))
vitesse_memoire = ti.Vector.field(2, dtype=ti.f32, shape=(Taille_memoire, Nbr_oiseaux))
frame_actuelle = ti.field(dtype=ti.i32, shape=())

# Paramètres de simulation variables (curseurs)
rayon_cohesion = ti.field(dtype=ti.f32, shape=())
rayon_alignement = ti.field(dtype=ti.f32, shape=())
rayon_repulsion = ti.field(dtype=ti.f32, shape=())

poid_cohesion = ti.field(dtype=ti.f32, shape=())
poid_alignement = ti.field(dtype=ti.f32, shape=())
poid_repulsion = ti.field(dtype=ti.f32, shape=())

nbr_oiseaux = ti.field(dtype=ti.i32, shape=())
vitesse_max = ti.field(dtype=ti.f32, shape=())
bruit = ti.field(dtype=ti.f32, shape=())
repulsion_mur = ti.field(dtype=ti.f32, shape=())


@ti.func
def taille_grille() -> mti.ivec2 :

    rayon_influence = mti.max(
        rayon_cohesion[None], 
        rayon_alignement[None], 
        rayon_repulsion[None]
        )
    
    taille = mti.ivec2(
        mti.floor(Hauteur/rayon_influence,ti.i32)+1,
        mti.floor(Largeur/rayon_influence,ti.i32)+1
        )
    
    return taille



@ti.kernel
def initialiser():
    # Initialisation des positions et vitesses des oiseaux
    for i in range(Nbr_oiseaux):
        position[i] = ti.Vector([ti.random * Largeur, ti.random() * Hauteur])
        angle = ti.random() * 2 * pi
        vitesse[i] = ti.Vector([mti.cos(angle), mti.sin(angle)]) # norme initiale ?????
        couleur[i] = ti.Vector([
            0.4 + ti.random() * 0.2,
            0.5 + ti.random() * 0.2,
            1.0
        ])



@ti.kernel
def deplacer():



def main ():
    # Initialisation des oiseaux
    initialiser()


if __name__ == "__main__":
    main()
