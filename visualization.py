import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as mpatch
import xml.etree.ElementTree as ET

yrblue = '#005396'
yrpink = '#cf006b'
yrgreen = '#539600'


def read_xml(in_file):
    tree = ET.parse(in_file)
    return tree


def get_subgrid_hierarchical(node, grid_dict, root_grid, color=False):
    dim = [int(node.attrib['dim0']), int(node.attrib['dim1'])]
    level = int(node.attrib['level'])
    sub_grid = [[None for i in range(dim[0])] for j in range(dim[1])]
    for i in range(dim[0]):
        for j in range(dim[1]):
            element = 'i{}j{}'.format(i, j)
            grid_dict[element] = {}
            sub_node = node.find(element)
            sub_type = sub_node.attrib['type']
            if sub_type == 'Hierarchical':
                grid_dict[element]['gs'] = gs.GridSpecFromSubplotSpec(
                    int(sub_node.attrib['dim0']),
                    int(sub_node.attrib['dim1']),
                    subplot_spec=root_grid[i, j]
                )
                grid_dict[element]['sub_gs'] = {}
                sub_grid[i][j] = get_subgrid_hierarchical(
                    sub_node,
                    grid_dict[element]['sub_gs'],
                    grid_dict[element]['gs'],
                    (level == 0 and [i, j] in [[0, 0], [0, 1]]) or color
                )
            elif sub_type == 'LowRank':
                # plot_lowrank(sub_node, grid_dict[element], root_grid[i, j])
                plot_lowrank_patches(
                    sub_node, grid_dict[element], root_grid[i, j], dim
                    # (level == 0 and [i, j] in [[0, 0], [0, 1]]) or (color and i >= j)
                )
            elif sub_type == 'Dense':
                # plot_dense(sub_node, grid_dict[element], root_grid[i, j])
                plot_dense_patch(
                    sub_node, grid_dict[element], root_grid[i, j]
                    # (level == 0 and [i, j] in [[0, 0], [0, 1]]) or (color and i >= j),
                    # 'lower'
                )
    return sub_grid


def plot_lowrank(xml_node, lowrank_dict, root_grid):
    dim = [int(xml_node.attrib['dim0']), int(xml_node.attrib['dim1'])]
    lowrank_dict['gs'] = gs.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=root_grid
    )
    svalues = [float(x)+1e-10 for x in xml_node.attrib['svalues'].split(',')]
    slog = np.log(svalues[0:min(10, *dim)])
    ax = plt.subplot(lowrank_dict['gs'][0, 0])
    # ax.text(
    #     0.5, 0.5,
    #     "{}\n({}, {})".format(
    #         xml_node.attrib['level'],
    #         xml_node.attrib['i_abs'], xml_node.attrib['j_abs']),
    #     horizontalalignment='center', verticalalignment='center'
    # )
    ax.set(
        xlim=(-0.5, len(slog)+0.5),
        ylim=(0, slog[0]-slog[-1]),
        xticks=[], yticks=[]
    )
    ax.bar(np.arange(len(slog)), slog-slog[-1], width=1, color=yrpink)


def plot_lowrank_patches(
    xml_node, lowrank_dict, root_grid, dim, has_color=True
):
    level = int(xml_node.attrib['level'])
    lowrank_dict['gs'] = gs.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=root_grid
    )
    ax = plt.subplot(lowrank_dict['gs'][0, 0])
    ax.set(xticks=[], yticks=[])
    dist = 0.0075*2**(level*np.sqrt(dim[0]/2)-int(level/4))
    k = (1.0 - 3*dist)/40*2**(level*np.sqrt(dim[0]/2)-int(level/4))
    n = (1.0 - 3*dist-k)
    color = yrpink if has_color else "grey"
    U = mpatch.Rectangle(
        (dist, dist),
        k, n,
        edgecolor='none',
        facecolor=color
    )
    S = mpatch.Rectangle(
        (dist, 2*dist+n),
        k, k,
        edgecolor='none',
        facecolor=color
    )
    V = mpatch.Rectangle(
        (2*dist+k, 2*dist+n),
        n, k,
        edgecolor='none',
        facecolor=color
    )
    # Add the patch to the Axes
    ax.add_patch(U)
    ax.add_patch(S)
    ax.add_patch(V)


def plot_dense(xml_node, dense_dict, root_grid):
    dim = [int(xml_node.attrib['dim0']), int(xml_node.attrib['dim1'])]
    dense_dict['gs'] = gs.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=root_grid
    )
    svalues = [float(x)+1e-10 for x in xml_node.attrib['svalues'].split(',')]
    slog = np.log(svalues[0:min(10, *dim)])
    ax = plt.subplot(dense_dict['gs'][0, 0])
    # ax.text(
    #     0.5, 0.5,
    #     "{}\n({}, {})".format(
    #         xml_node.attrib['level'],
    #         xml_node.attrib['i_abs'], xml_node.attrib['j_abs']),
    #     horizontalalignment='center', verticalalignment='center'
    # )
    ax.set(
        xlim=(-0.5, len(slog)+0.5),
        ylim=(0, slog[0]-slog[-1]),
        xticks=[], yticks=[]
    )
    ax.bar(np.arange(len(slog)), slog-slog[-1], width=1, color=yrblue)


def plot_dense_patch(xml_node, dense_dict, root_grid, has_color=True, ul=None):
    dense_dict['gs'] = gs.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=root_grid
    )
    ax = plt.subplot(dense_dict['gs'][0, 0])
    ax.set(xticks=[], yticks=[])
    color = yrblue if has_color else "grey"
    if ul is None or not has_color:
        patch = mpatch.Rectangle(
            (0.0, 0.0),
            1.0, 1.0,
            edgecolor='none',
            facecolor=color
        )
        # Add the patch to the Axes
        ax.add_patch(patch)
    else:
        upper = mpatch.Polygon(
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            edgecolor='none',
            facecolor=color if ul == 'upper' else 'grey'
        )
        ax.add_patch(upper)
        lower = mpatch.Polygon(
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            edgecolor='none',
            facecolor=color if ul == 'lower' else 'grey'
        )
        ax.add_patch(lower)


def plot_matrix(root):
    grid_dict = {}
    grid_dict['root'] = {}
    if root.attrib['type'] == 'Hierarchical':
        grid_dict['root']['gs'] = gs.GridSpec(
            int(root.attrib['dim0']),
            int(root.attrib['dim1'])
        )
        grid_dict['root']['sub_gs'] = {}
        get_subgrid_hierarchical(
            root,
            grid_dict['root']['sub_gs'],
            grid_dict['root']['gs']
        )
    return grid_dict


def main():
    in_file = 'matrix.xml' if len(sys.argv) == 1 else sys.argv[1]
    tree = read_xml(in_file)
    root = tree.getroot()

    plt.figure(figsize=(5, 5), dpi=200)
    plot_matrix(root)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("matrix.pdf")
    # plt.show()


if __name__ == '__main__':
    main()
