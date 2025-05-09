import taichi as ti

ti.init(arch=ti.gpu)

radius_field = ti.field(dtype=ti.f32, shape=())
color_field = ti.Vector.field(3, dtype=ti.f32, shape=())

@ti.kernel
def do_something():
    r = radius_field[None]
    intensity = r / 50.0
    color_field[None] = ti.Vector([intensity, 0.2, 1.0 - intensity])

def rgb_to_hex(c):
    r = int(c[0] * 255)
    g = int(c[1] * 255)
    b = int(c[2] * 255)
    return (r << 16) + (g << 8) + b

def main():
    gui = ti.GUI('Radius Color Circle', background_color=0xFFFFFF)
    radius = gui.slider('Radius', 1, 50, step=1)

    radius.value = 10
    radius_field[None] = radius.value
    color_field[None] = ti.Vector([1.0, 0.0, 0.0])

    while gui.running:
        radius_field[None] = radius.value
        do_something()

        color = rgb_to_hex(color_field[None])

        gui.circle((0.5, 0.5), radius=radius.value, color=color)
        gui.show()

if __name__ == '__main__':
    main()