import math


class Proj:
    """Orthographic 3/4 view. World: x east, y north (away), h up. Viewer is south-east, above.

    Visible outer faces of a box are its south (y=min) and east (x=max) faces.
    """

    def __init__(self, ox, oy, az=28.0, el=34.0):
        self.az = math.radians(az)
        e = math.radians(el)
        self.ox, self.oy = ox, oy
        self.ca, self.sa = math.cos(self.az), math.sin(self.az)
        self.ce, self.se = math.cos(e), math.sin(e)

    def depth(self, x, y):
        """Larger = farther from the viewer."""
        return -x * self.sa + y * self.ca

    def __call__(self, x, y, h=0.0):
        sx = x * self.ca + y * self.sa
        sy = -h * self.ce - self.depth(x, y) * self.se
        return (self.ox + sx, self.oy + sy)
