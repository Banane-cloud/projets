import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon

Nb=70
longueur = 200
largeur =200
hauteur = 200
position =np.zeros((Nb,3))
vitesse =np.zeros((Nb,3))
couleur =np.zeros((Nb,3))
Vmax=30

rayon_cohesion = 50
rayon_alignement = 15
rayon_repulsion = 10

poid_cohesion = 4
poid_alignement = 1
poid_repulsion = 80
poids_mur = 150
angle_vision = 120

bruit = 30
Masse =1



def initialiser():
    for i in range (Nb):
        position[i] = np.random.random()*longueur, np.random.random()*largeur, np.random.random()*hauteur
        vitesse[i] = np.random.uniform(-Vmax, Vmax, 3)
        couleur[i] = np.random.random(), np.random.random(), np.random.random()

def avancer (dt): 
    for i in range (Nb):
        position[i] += vitesse[i]*dt
        bordure2(i)

def bordure (i):
    if position[i][0] > longueur:
        position[i][0] = 0     
    if position[i][0] < 0:
        position[i][0] = longueur   
    if position[i][1] > largeur:
        position[i][1] = 0
    if position[i][1] < 0:
        position[i][1] = largeur
    if position[i][2] > hauteur:
        position[i][2] = 0
    if position[i][2] < 0:
        position[i][2] = hauteur

def bordure2 (i):
    if position[i][0] > longueur:
        position[i][0] *= -1
    if position[i][0] < 0:
        position[i][0] *= -1
    if position[i][1] > largeur:
        position[i][1] *= -1
    if position[i][1] < 0:
        position[i][1] *= -1
    if position[i][2] > hauteur:
        position[i][2] *= -1
    if position[i][2] < 0:
        position[i][2] *= -1



def accelerer(dt):
    for i in range (Nb):
        vitesse[i] += (force(i)+bruit*np.random.uniform(-1,1,3))*dt/Masse

        # Limiter la vitesse à Vmax
        vitesse_magnitude = np.linalg.norm(vitesse[i])
        if vitesse_magnitude > Vmax:
            vitesse[i] = (vitesse[i] / vitesse_magnitude) * Vmax  # Normalisation et application de la vitesse maximale

def force(i):
    force = np.zeros((3))
    force += force_cohesion(i)
    force += force_alignement(i)
    force += force_repulsion(i)
    force += force_mur(i)
    return (force)

def force_cohesion(i):
    force=np.zeros((3))
    centre_masse = np.zeros((3))
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
    force=np.zeros((3))
    vitesse_moyenne = np.zeros((3))
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
    force=np.zeros((3))

    for j in range (Nb):
        if i != j and distance(i,j) < rayon_repulsion and vision(i,j):
            force += (position[i] - position[j]) / (np.linalg.norm(position[i] - position[j]) ** 2)
    return (force*poid_repulsion)

def force_mur(i):
    force = np.zeros((3))
    delta = min(longueur, largeur, hauteur) / 4  # Distance de sécurité pour les murs
    if position[i][0] < delta :
        force[0] = 1/np.abs(position[i][0])*poids_mur
    if position[i][0] > longueur - delta:
        force[0] = -1/np.abs(position[i][0]-longueur)*poids_mur
    if position[i][1] < delta:
        force[1] = 1/np.abs(position[i][1])*poids_mur
    if position[i][1] > largeur - delta:
        force[1] = -1/np.abs(position[i][1]-largeur)*poids_mur
    if position[i][2] < delta:
        force[2] = 1/np.abs(position[i][2])*poids_mur
    if position[i][2] > hauteur - delta:
        force[2] = -1/np.abs(position[i][2]-hauteur)*poids_mur
    return force
        

def distance(i,j):
    x = position[j][0] - position[i][0]
    y = position[j][1] - position[i][1]
    z= position[j][2] - position[i][2]
    return np.sqrt(x**2 + y**2 + z**2)

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
    dt = 0.1

    # Initialiser la figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, longueur)
    ax.set_ylim(0, largeur)
    ax.set_zlim(0, hauteur)

    # Création du scatter une seule fois
    scat = ax.scatter(position[:, 0], position[:, 1], position[:, 2], c=couleur)

    def animate(i):
        accelerer(dt)
        avancer(dt)

        # Met à jour les données du scatter 3D
        scat._offsets3d = (position[:, 0], position[:, 1], position[:, 2])
        return scat,

    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    main()