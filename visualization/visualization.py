import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import xml.etree.ElementTree as ET


def read_xml():
    tree = ET.parse('input/test.xml');
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
                grid_dict[element]['gs'] = gs.GridSpecFromSubplotSpec(
                    1, 1,
                    subplot_spec=root_grid[i, j]
                )
                ax = plt.subplot(grid_dict[element]['gs'][0, 0])
                ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
                ax.plot([0,1], [0, 1], color='r')
            elif sub_type == 'Dense':
                grid_dict[element]['gs'] = gs.GridSpecFromSubplotSpec(
                    1, 1,
                    subplot_spec=root_grid[i, j]
                )
                ax = plt.subplot(grid_dict[element]['gs'][0, 0])
                ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
                ax.plot([0,1], [1, 0], color='b')
    return sub_grid


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
    plt.show()

    # root_gs = gs.GridSpec(2, 2)
    # sub_gs = [[None for i in range(2)] for j in range(2)]
    # for i in range(2):
    #     for j in range(2):
    #         sub_gs[i][j] = gs.GridSpecFromSubplotSpec(
    #             2, 2,
    #             subplot_spec=root_gs[i,j],
    #             hspace=0.0, wspace=0.0
    #         )

    # print(sub_gs)
    # for m in range(2):
    #     for n in range(2):
    #         for i in range(2):
    #             for j in range(2):
    #                 ax = plt.subplot(sub_gs[m][n][i,j])
    #                 ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    #                 ax.plot([0,1], [m == n, m != n], color='r')
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    # fig, axs = plt.subplots(5, 5)
    # for ax in axs.flat:
    #     ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    # fig.tight_layout(pad=0)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()


if __name__ == '__main__':
    main()
