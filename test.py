r = 0.4
g = 0.5
b = 1.0

color = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
print(hex(color))