import casadi as ca
from casadi.tools.graph import graph
import matplotlib.image as mpimg
from io import BytesIO
import matplotlib.pyplot as plt
import os


def draw_graph(x):
    g = graph.dotgraph(x)
    g.set('dpi', 300)
    png = g.create('dot', 'png')
    bio = BytesIO()
    bio.write(png)
    bio.seek(0)
    img = mpimg.imread(bio)
    bio.close()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    if os.path.isfile('source.dot'):
        os.remove('source.dot')


x = ca.SX.sym('x')
draw_graph(2*x + 1)