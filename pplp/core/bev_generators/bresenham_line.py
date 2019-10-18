#!/usr/bin/env python

# def get_line(start, end, dimX, dimY):
#     """Bresenham's Line Algorithm
#     Produces a list of tuples of cells on the ray though start and end.
#     """
#     # Setup initial conditions
#     x1, y1 = start
#     x2, y2 = end
#     dx = x2 - x1
#     dy = y2 - y1
#
#     # Determine how steep the line is
#     is_steep = abs(dy) > abs(dx)
#
#     # Rotate line
#     if is_steep:
#         dimX, dimY = dimY, dimX
#         x1, y1 = y1, x1
#         x2, y2 = y2, x2
#
#     # Swap start and end points if necessary and store swap state
#     swapped = False
#     if x1 > x2:
#         x1, x2 = x2, x1
#         y1, y2 = y2, y1
#         swapped = True
#
#     # Recalculate differentials
#     dx = x2 - x1
#     dy = y2 - y1
#
#     # Calculate error
#     error = int(dx / 2.0)
#     ystep = 1 if y1 < y2 else -1
#
#     # Iterate over bounding box generating points between start and end
#     y = y1
#     points = []
#
#     for x in range(x1, dimX):
#         if y >= dimY - 1 or y < 0:
#             break
#         coord = (y, x) if is_steep else (x, y)
#         points.append(coord)
#         error -= abs(dy)
#         if error < 0:
#             y += ystep
#             error += dx
#
#     # Reverse the list if the coordinates were swapped
#     if swapped:
#         points.reverse()
#     return points


# Denser than get_line
def supercover_line(start, end, dimX, dimY):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x = x0
    y = y0
    x_inc = 1 if (x1 > x0) else -1
    y_inc = 1 if (y1 > y0) else -1
    error = dx - dy
    dx *= 2
    dy *= 2

    points = []
    for n in reversed(range(0, 1 + dimX + dimY)):

        points.append((x, y))

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

        if y >= dimY - 1 or y < 0:
            break
        if x >= dimX - 1 or x < 0:
            break

    return points

# Not good algorithm
# def DrawLine(start, end, dimX, dimY, fThick=True):
#     yStep = xStep = 1    # the step on y and x axis
#     Error = 0           # the error accumulated during the increment
#     x0, y0 = start
#     x1, y1 = end
#     x = x0
#     y = y0
#     dx = x1 - x0
#     dy = y1 - y0
#
#     points = []
#
#     points.append((x, y)) # First point
#
#     if fThick:
#         points.append((x+1, y+1))
#
#     if dy < 0:
#         yStep = -1
#         dy = -dy
#     else:
#         yStep = 1
#
#     if dx < 0:
#         xStep = -1
#         dx = -dx
#     else:
#         xStep = 1
#
#     ddy = 2 * dy
#     ddx = 2 * dx
#
#     if ddx >= ddy:                 # first octant (0 <= slope <= 1)
#         Error = dx                 # start in the middle of the square
#         for i in range(dimX):    # do not use the first po(already done)
#             x += xStep
#             Error += ddy
#             if Error > ddx:        # increment y if AFTER the middle (>)
#                 y += yStep
#                 Error -= ddx
#             if y >= dimY - 1 or y < 0:
#                 break
#             points.append((x, y))
#             if fThick:
#                 if 0 < y+xStep < dimY - 1 and 0 < x+yStep < dimX - 1:
#                     points.append((x+yStep, y+xStep))
#     else:
#         Error = dy
#         for i in range(dimY):
#             y += yStep
#             Error += ddx
#             if Error > ddy:
#                 x += xStep
#                 Error -= ddy
#             if x >= dimX - 1 or x < 0:
#                 break
#             points.append((x, y))
#             if fThick:
#                 if 0 < y + xStep < dimY - 1 and 0 < x + yStep < dimX - 1:
#                     points.append((x+yStep, y+xStep))
#     return points


# if __name__ == '__main__':
#     points = raytrace((394, -1), (148, 435), 700, 800)
#     # points = DrawLine((0, 1), (6, 4), 700, 800)
#     print(len(points))
#
#     for (x, y) in points:
#         print(x, y)



