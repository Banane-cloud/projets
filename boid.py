## Simulation interraction d'oiseaux via le modèle de Boids ##

# Importation des bibliothèques nécessaires
import taichi as ti
import taichi.math as mti
from math import pi 
import time

# Initialisation de Taichi avec l'architecture GPU
ti.init(arch=ti.gpu)

# gestion de la simulation
pause = ti.field(dtype=ti.i32, shape=())

# Définition des paramètres de simulation
Hauteur, Largeur = 1000, 2000
Taille_memoire = 200
nbr_oiseaux = ti.field(dtype=ti.i32, shape=())

# Définition des champs de positions et de vitesses et de couleur des oiseaux∏
position = ti.Vector.field(2, dtype=ti.f32, shape=nbr_oiseaux[None])
vitesse = ti.Vector.field(2, dtype=ti.f32, shape=nbr_oiseaux[None])
couleur = ti.Vector.field(3, dtype=ti.f32, shape=nbr_oiseaux[None])

# Définition des champs de mémoire pour les oiseaux
position_memoire = ti.Vector.field(2, dtype=ti.f32, shape=(Taille_memoire, nbr_oiseaux[None]))
vitesse_memoire = ti.Vector.field(2, dtype=ti.f32, shape=(Taille_memoire, nbr_oiseaux[None]))
frame_actuelle = ti.field(dtype=ti.i32, shape=())

# Paramètres de simulation variables (curseurs)
rayon_cohesion = ti.field(dtype=ti.f32, shape=())
rayon_alignement = ti.field(dtype=ti.f32, shape=())
rayon_repulsion = ti.field(dtype=ti.f32, shape=())

poid_cohesion = ti.field(dtype=ti.f32, shape=())
poid_alignement = ti.field(dtype=ti.f32, shape=())
poid_repulsion = ti.field(dtype=ti.f32, shape=())

vitesse_max = ti.field(dtype=ti.f32, shape=())
angle_vision = ti.field(dtype=ti.f32, shape=())
bruit = ti.field(dtype=ti.f32, shape=())

# Paramètres de l'environnement
repulsion_mur = ti.field(dtype=ti.f32, shape=())


@ti.func
def taille_grille() -> mti.ivec2:

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
    for i in range(nbr_oiseaux[None]):
        position[i] = ti.Vector([ti.random * Largeur, ti.random() * Hauteur])
        angle = ti.random() * 2 * pi
        vitesse[i] = ti.Vector([mti.cos(angle), mti.sin(angle)]) # norme initiale ?????
        couleur[i] = ti.Vector([
            0.4 + ti.random() * 0.2,
            0.5 + ti.random() * 0.2,
            1.0
        ])




@ti.kernel
def deplacer(dt: float):
    A=1


def main ():
    # Initialisation de l'interface graphique

    gui = ti.GUI("Simulation de Boids", (Largeur, Hauteur), background_color=0xADD8E6)

    nbr_oiseaux[None] = 100
    rayon_cohesion[None] = 0.3
    rayon_alignement[None] = 0.2
    rayon_repulsion[None] = 0.1

    poid_cohesion[None] = 0.5
    poid_alignement[None] = 0.5
    poid_repulsion[None] = 0.5

    vitesse_max[None] = 10
    angle_vision[None] = 360
    bruit[None] = 0.1

    repulsion_mur[None] = 0.1
    pause[None] = 1

    dt_min = 1/60

    # Initialisation des oiseaux
    initialiser()



    while gui.running:
        
        # gestion des FPS
        temp_act = time.time()
        dt=temp_act - temps_prec
        if dt < dt_min:
            time.sleep(dt_min - dt)
            dt = dt_min
        temps_prec=temp_act

        # Gestion graphique
        

        # Gestion des événements de la GUI
        for event in gui.get_events(gui.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == ti.GUI.SPACE:
                # Mettre en pause la simulation
                if pause[None] == 0:
                    pause[None] = 1
                else:
                    pause[None] = 0
            elif event.key == ti.GUI.LEFT and pause[None] == 1:
                # Reculer dans l'historique
                A=1
            elif event.key == ti.GUI.RIGHT and pause[None] == 1:
                # Avancer dans l'historique ou avancer la simulation
                A=1
            elif event.key == 'r':
                initialiser()


        # Déplacement des oiseaux
        if pause[None] == 0:
            deplacer(dt)




if __name__ == "__main__":
    main()
