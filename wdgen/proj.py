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


def sort_boxes(P, items, eps=0.25):
    """Topologically sort `(box, payload)` items back-to-front for orthographic 3/4 view (+x east, +y north).

    Viewer is south-east (+x, -y), so box A=(ax0, ay0, ax1, ay1) must be drawn BEFORE B=(bx0, by0, bx1, by1)
    whenever B can occlude A:
      - A is west of B (ax1 <= bx0 + eps) and not strictly south of B (ay1 > by0 + eps), OR
      - A is north of B (by1 <= ay0 + eps) and not strictly east of B (ax0 < bx1 - eps).
    Ties (and any rare cycle from overlapping footprints) break by descending P.depth(cx, cy).
    Returns list of payloads in back-to-front draw order.
    """
    n = len(items)
    if n <= 1:
        return [it[1] for it in items]
    boxes = [it[0] for it in items]
    depths = [P.depth((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5) for b in boxes]
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = boxes[j]
            i_before_j = (ax1 <= bx0 + eps and ay1 > by0 + eps) or (by1 <= ay0 + eps and ax0 < bx1 - eps)
            j_before_i = (bx1 <= ax0 + eps and by1 > ay0 + eps) or (ay1 <= by0 + eps and bx0 < ax1 - eps)
            if i_before_j and not j_before_i:
                adj[i].append(j)
                indeg[j] += 1
            elif j_before_i and not i_before_j:
                adj[j].append(i)
                indeg[i] += 1
    remaining = set(range(n))
    order = []
    while remaining:
        ready = [i for i in remaining if indeg[i] == 0]
        u = max(ready if ready else remaining, key=lambda idx: depths[idx])
        remaining.remove(u)
        order.append(items[u][1])
        for v in adj[u]:
            if v in remaining:
                indeg[v] -= 1
    return order
