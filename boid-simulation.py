import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon

Nb=50
hauteur = 200
largeur =200
position =np.zeros((Nb,2))
vitesse =np.zeros((Nb,2))
couleur =np.zeros((Nb,3))
Vmax=30

rayon_cohesion = 30
rayon_alignement = 15
rayon_repulsion = 10

poid_cohesion = 20
poid_alignement = 5
poid_repulsion = 200
angle_vision = 120

bruit = 15
Masse =5


def initialiser():
    for i in range (Nb):
        position[i] = np.random.random()*largeur, np.random.random()*hauteur
        vitesse[i] = np.random.uniform(-Vmax, Vmax, 2)
        couleur[i] = np.random.random(), np.random.random(), np.random.random()

def avancer (dt): 
    for i in range (Nb):
        position[i] += vitesse[i]*dt
        bordure(i)

def bordure (i):
    if position[i][0] > largeur:
        position[i][0] = 0     
    if position[i][1] > hauteur:
        position[i][1] = 0
    if position[i][0] < 0:
        position[i][0] = largeur
    if position[i][1] < 0:
        position[i][1] = hauteur

def accelerer(dt):
    for i in range (Nb):
        vitesse[i] += (force(i)+bruit*np.random.uniform(-1,1,2))*dt/Masse

        # Limiter la vitesse à Vmax
        vitesse_magnitude = np.linalg.norm(vitesse[i])
        if vitesse_magnitude > Vmax:
            vitesse[i] = (vitesse[i] / vitesse_magnitude) * Vmax  # Normalisation et application de la vitesse maximale

def force(i):
    force = np.zeros((2))
    force += force_cohesion(i)
    force += force_alignement(i)
    force += force_repulsion(i)
    return (force)

def force_cohesion(i):
    force=np.zeros((2))
    centre_masse = np.zeros((2))
    voisins = 0

    for j in range (Nb):
        if i != j and distance(i,j) < rayon_cohesion and vision(i,j):
            centre_masse += position[j]
            voisins += 1
    if voisins > 0:
        centre_masse /= voisins
        force = poid_cohesion * (centre_masse - position[i])
    return (force)

def force_alignement(i):
    force=np.zeros((2))
    vitesse_moyenne = np.zeros((2))
    voisins = 0

    for j in range (Nb):
        if i != j and distance(i,j) < rayon_alignement and vision(i,j):
            vitesse_moyenne += vitesse[j]
            voisins += 1
    if voisins > 0:
        vitesse_moyenne /= voisins
        force = poid_alignement * (vitesse_moyenne - vitesse[i])
    return (force)

def force_repulsion(i):
    force=np.zeros((2))

    for j in range (Nb):
        if i != j and distance(i,j) < rayon_repulsion and vision(i,j):
            force += (position[i] - position[j]) / (np.linalg.norm(position[i] - position[j]) ** 2)
    return (force*poid_repulsion)


def distance(i,j):
    x = position[j][0] - position[i][0]
    y = position[j][1] - position[i][1]
    return np.sqrt(x**2 + y**2)

def vision(i,j):
    direction = vitesse[i]
    to_j = position[j] - position[i]

    norm_dir = np.linalg.norm(direction)
    norm_to_j = np.linalg.norm(to_j)

    if norm_dir == 0 or norm_to_j == 0:
        return False

    cos_theta = np.clip(np.dot(direction, to_j) / (norm_dir * norm_to_j), -1.0, 1.0)
    angle = np.arccos(cos_theta)  # radians

    return abs(np.degrees(angle)) < angle_vision / 2



def main (): 
    initialiser()
    dt=0.1
    # Figure et scatter
    fig, ax = plt.subplots()
    
    scat = ax.scatter(position[:, 0], position[:, 1], c=couleur)
    ax.set_xlim(0, largeur)
    ax.set_ylim(0, hauteur)

    def animate(i):
        accelerer(dt)
        avancer(dt)
        scat.set_offsets(position)
        return scat,

    ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()