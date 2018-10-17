import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as mpatch
import matplotlib.collections as mcoll
import xml.etree.ElementTree as ET


def read_xml():
    tree = ET.parse('matrix.xml');
    return tree


def get_subgrid_hierarchical(node, grid_dict, root_grid):
    dim = [int(node.attrib['dim0']), int(node.attrib['dim1'])]
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
                )
            elif sub_type == 'LowRank':
                plot_lowrank(sub_node, grid_dict[element], root_grid[i, j])
            elif sub_type == 'Dense':
                plot_dense(sub_node, grid_dict[element], root_grid[i, j])
    return sub_grid


def plot_lowrank(xml_node, lowrank_dict, root_grid):
    dim = [int(xml_node.attrib['dim0']), int(xml_node.attrib['dim1'])]
    rank = int(xml_node.attrib['rank'])
    lowrank_dict['gs'] = gs.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=root_grid
    )
    svalues = [float(x)+1e-10 for x in xml_node.attrib['svalues'].split(',')]
    slog = np.log(svalues[0:10])
    ax = plt.subplot(lowrank_dict['gs'][0, 0])
    ax.set(xlim=(0, len(slog)), ylim=(0, slog[0]-slog[9]), xticks=[], yticks=[])
    ax.bar(np.arange(len(slog)), slog-slog[9], width=1, color='r')


def plot_dense(xml_node, dense_dict, root_grid):
    dim = [int(xml_node.attrib['dim0']), int(xml_node.attrib['dim1'])]
    dense_dict['gs'] = gs.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=root_grid
    )
    svalues = [float(x)+1e-10 for x in xml_node.attrib['svalues'].split(',')]
    slog = np.log(svalues[0:10])
    ax = plt.subplot(dense_dict['gs'][0, 0])
    ax.set(xlim=(0, len(slog)), ylim=(0, slog[0]-slog[9]), xticks=[], yticks=[])
    ax.bar(np.arange(len(slog)), slog-slog[9], width=1, color='b')


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
    tree = read_xml()
    root = tree.getroot()

    fig = plt.figure()
    grid_dict = plot_matrix(root)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("matrix.pdf")
    plt.show()


if __name__ == '__main__':
    main()
