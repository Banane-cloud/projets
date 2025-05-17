import taichi as ti

ti.init(arch=ti.metal)

Nmax= 5000
Nboids = ti.field(dtype=ti.i32, shape=())
Nboids[None] = 1000
Nflocks = ti.field(dtype=ti.i32, shape=())
Nflocks[None] = 3

palette = [
    0xE63946,  # rouge framboise
    0xF28E2B,  # orange foncé
    0xF1C40F,  # jaune profond
    0x90BE6D,  # vert olive clair
    0x43AA8B,  # vert forêt doux
    0x3FB8AF,  # turquoise vibrant
    0x2C7BB6,  # bleu océan
    0x4D4EEC,  # bleu indigo
    0x9B51E0,  # violet moyen
    0xD64DB7   # rose framboise
]

grid_size = ti.Vector.field(2, dtype=ti.i32, shape=())

positions = ti.Vector.field(2, dtype=ti.f32, shape=Nmax)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=Nmax)
noise_vector = ti.Vector.field(2, dtype=ti.f32, shape=Nmax)
flock = ti.field(dtype=ti.i32, shape=Nmax)


# Parameters
Longueur = 1400
Largeur = 800
# Hauteur = 100

Vmax = ti.field(dtype=ti.f32, shape=())
Vmax[None] = 30

gravity = ti.Vector.field(2,dtype=ti.f32, shape=())
gravity[None] = ti.Vector([0, 0])

noise = ti.field(dtype=ti.f32, shape=())
noise[None] = 0
noise_max = 10

cohesion_radius = ti.field(dtype=ti.f32, shape=())
cohesion_radius[None] = 30
alignment_radius = ti.field(dtype=ti.f32, shape=())
alignment_radius[None] = 15
repulsion_radius = ti.field(dtype=ti.f32, shape=())
repulsion_radius[None] = 10

cohesion_strength = ti.field(dtype=ti.f32, shape=())
cohesion_strength[None] = 1
alignment_strength = ti.field(dtype=ti.f32, shape=())
alignment_strength[None] = 1
repulsion_strength = ti.field(dtype=ti.f32, shape=())
repulsion_strength[None] = 1

vision_angle = ti.field(dtype=ti.f32, shape=())
vision_angle[None] = 120


@ti.kernel
def init_boids():
    """
    Initialize boids with random positions and velocities.
    """
    for i in range(Nmax):
        positions[i] = ti.Vector([ti.random()*Longueur, ti.random()*Largeur])
        velocities[i] = ti.Vector([(ti.random()*2 - 1) * Vmax[None], (ti.random()*2 - 1) * Vmax[None]])
        flock[i] = int(ti.random() * Nflocks[None])


@ti.kernel
def bordure():
    """
    Handle the border conditions for boids.
    """
    for i in range(Nboids[None]):
        if positions[i][0] > Longueur:
            positions[i][0] = 0     
        if positions[i][1] > Largeur:
            positions[i][1] = 0
        if positions[i][0] < 0:
            positions[i][0] = Longueur
        if positions[i][1] < 0:
            positions[i][1] = Largeur

@ti.kernel
def update(dt : float):
    for i in range(Nboids[None]):
        positions[i] += velocities[i] * dt

@ti.kernel
def grid():
    """
    Create a grid for the boids. useless for now
    """
    max_radius = max(cohesion_radius[None], alignment_radius[None], repulsion_radius[None])
    # Calculate the number of grid cells
    grid_size[None] = int (Longueur / max_radius),int (Largeur / max_radius)

    for i in range(Nboids[None]):
        # Calculate the grid cell for each boid
        x = int(positions[i][0] / grid_size[None][0])
        y = int(positions[i][1] / grid_size[None][1])


@ti.func
def compute_all_forces(i: int) -> ti.Vector:
    pos_mean = ti.Vector([0.0, 0.0])
    vel_mean = ti.Vector([0.0, 0.0])
    repulsion = ti.Vector([0.0, 0.0])
    count_cohesion = 0
    count_alignment = 0

    for j in range(Nboids[None]):
        if i == j:
            continue

        offset = positions[j] - positions[i]
        dist2 = offset.norm_sqr()
        in_vision = vision(i, j)

        # Repulsion (indépendante du flock)
        if dist2 < repulsion_radius[None] * repulsion_radius[None] and in_vision:
            repulsion += offset / dist2

        # Cohésion & alignement : seulement si même flock
        if flock[i] != flock[j]:
            continue

        if dist2 < cohesion_radius[None] * cohesion_radius[None] and in_vision:
            pos_mean += positions[j]
            count_cohesion += 1

        if dist2 < alignment_radius[None] * alignment_radius[None] and in_vision:
            vel_mean += velocities[j]
            count_alignment += 1

    force = ti.Vector([0.0, 0.0])

    if count_cohesion > 0:
        pos_mean /= count_cohesion
        force += (pos_mean - positions[i]) * cohesion_strength[None]

    if count_alignment > 0:
        vel_mean /= count_alignment
        force += (vel_mean - velocities[i]) * alignment_strength[None]

    force += -repulsion * repulsion_strength[None]

    return force

@ti.kernel
def generate_noise():
    for i in range(Nboids[None]):
        noise_vector[i] = ti.Vector([
            (ti.random() * 2 - 1) * noise_max,
            (ti.random() * 2 - 1) * noise_max
        ])


@ti.func
def vision(i : int, j : int) -> bool:
    """
    Check if boid i can see boid j.
    """
    direction = velocities[i]
    to_j = positions[j] - positions[i]

    norm_dir = direction.norm()
    norm_to_j = to_j.norm()
    vision = False

    if norm_dir == 0 or norm_to_j == 0:
        vision = False
    else:
        angle = ti.acos(direction.dot(to_j) / (norm_dir * norm_to_j))
        if angle < (vision_angle[None] / 2) * (3.14 / 180):
            vision = True
    
    return vision

@ti.kernel
def accelerate(dt : float):
    """
    Update the velocities of the boids based on their interactions.
    """

    #compute the forces for each boid
    for i in range(Nboids[None]):
        
    #     pos_mean = ti.Vector([0.0, 0.0])
    #     vel_mean = ti.Vector([0.0, 0.0])
    #     repulsion = ti.Vector([0.0, 0.0])
    #     count_cohesion = 0
    #     count_alignment = 0

    #     for j in range(Nboids[None]):
    #         if i == j:
    #             continue

    #         offset = positions[j] - positions[i]
    #         dist2 = offset.norm_sqr()
    #         in_vision = vision(i, j)

    #         # Repulsion (indépendante du flock)
    #         if dist2 < repulsion_radius[None] * repulsion_radius[None] and in_vision:
    #             repulsion += offset / dist2

    #         # Cohésion & alignement : seulement si même flock
    #         if flock[i] != flock[j]:
    #             continue

    #         if dist2 < cohesion_radius[None] * cohesion_radius[None] and in_vision:
    #             pos_mean += positions[j]
    #             count_cohesion += 1

    #         if dist2 < alignment_radius[None] * alignment_radius[None] and in_vision:
    #             vel_mean += velocities[j]
    #             count_alignment += 1

    #     force = ti.Vector([0.0, 0.0])

    #     if count_cohesion > 0:
    #         pos_mean /= count_cohesion
    #         force += (pos_mean - positions[i]) * cohesion_strength[None]

    #     if count_alignment > 0:
    #         vel_mean /= count_alignment
    #         force += (vel_mean - velocities[i]) * alignment_strength[None]

    #     force += -repulsion * repulsion_strength[None]
        
        # Apply the forces to the boid
        force = compute_all_forces(i)

        velocities[i] += gravity[None] * dt
        velocities[i] += force * dt
        velocities[i] += noise[None] * noise_vector[i] * dt
        
        # Limit the speed to Vmax
        speed = velocities[i].norm()
        if speed > Vmax[None]:
            velocities[i] = (velocities[i] / speed) * Vmax[None]



def main() : 
    gui = ti.GUI('Boids', res=(Longueur, Largeur), background_color=0xADD8E6)
    init_boids()
    dt=0.05

    Nboids_slider = gui.slider('Number of Boids', 100, Nmax, step=1)
    Nboids_slider.value = Nboids[None]

    gravity_slider = gui.slider('Gravity', -10, 10, step=0.1)
    gravity_slider.value = 0

    noise_slider = gui.slider('Noise', 0, 10, step=0.1)
    noise_slider.value = 10

    Vmax_slider = gui.slider('Max Speed', 0, 100, step=1)
    Vmax_slider.value = 30

    vision_angle_slider = gui.slider('Vision Angle', 0, 360, step=1)
    vision_angle_slider.value = 120

    cohesion_radius_slider = gui.slider('Cohesion Radius', 0, 100, step=1)
    cohesion_radius_slider.value = 30

    cohesion_strength_slider = gui.slider('Cohesion Strength', 0, 100, step=1)
    cohesion_strength_slider.value = 0.05

    alignment_radius_slider = gui.slider('Alignment Radius', 0, 100, step=1)
    alignment_radius_slider.value = 15

    alignment_strength_slider = gui.slider('Alignment Strength', 0, 100, step=1)
    alignment_strength_slider.value = 0.1

    repulsion_radius_slider = gui.slider('Repulsion Radius', 0, 100, step=1)
    repulsion_radius_slider.value = 10

    repulsion_strength_slider = gui.slider('Repulsion Strength', 0, 5000, step=1)
    repulsion_strength_slider.value = 10


    paused = False

    while gui.running:
        # Handle events
        for e in gui.get_events():
            if e.key == ti.GUI.ESCAPE:
                gui.running = False  
            elif e.key == ti.GUI.SPACE and e.type == ti.GUI.PRESS:
                paused = not paused  


        # Updtae the parameters from the sliders
        gravity[None][1] = gravity_slider.value
        noise[None] = noise_slider.value
        cohesion_radius[None] = cohesion_radius_slider.value
        cohesion_strength[None] = cohesion_strength_slider.value
        alignment_radius[None] = alignment_radius_slider.value
        alignment_strength[None] = alignment_strength_slider.value
        repulsion_radius[None] = repulsion_radius_slider.value
        repulsion_strength[None] = repulsion_strength_slider.value
        Vmax[None] = Vmax_slider.value
        vision_angle[None] = vision_angle_slider.value
        Nboids[None] = int(Nboids_slider.value)


        if not paused:
            update(dt)
            generate_noise()
            accelerate(dt)
            bordure()

        n = Nboids[None]  # Nombre actuel de boids

        normalized_positions = positions.to_numpy()[:n].copy()
        normalized_positions[:, 0] /= Longueur
        normalized_positions[:, 1] /= Largeur

        gui.circles(normalized_positions, radius=2, palette=palette, palette_indices=flock.to_numpy()[:n])
        gui.show()

if __name__ == "__main__":
    main()
    




        



    
