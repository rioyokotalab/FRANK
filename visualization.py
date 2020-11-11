import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as mpatch
import xml.etree.ElementTree as ET

yrblue = '#005396'
yrpink = '#cf006b'
yrgreen = '#539600'
num_singular_values = 20
tolerance = 1e-5


def read_xml(in_file):
    tree = ET.parse(in_file)
    return tree


def plot_hierarchical(node, gs_node, color=False):
    dim = [int(node.attrib['dim0']), int(node.attrib['dim1'])]
    level = int(node.attrib['level'])
    shared_col_b = [
        {
            'exists':False, 'children':None,
            'level':level+1, 'dim':dim[0],
            'width': None
        }
        for i in range(dim[0])
    ]
    shared_row_b = [
        {
            'exists':False, 'children':None,
            'level':level+1, 'dim':dim[1],
            'width': None
        }
        for j in range(dim[1])
    ]
    gs_subgrid = gs.GridSpecFromSubplotSpec(
        dim[0], dim[1], subplot_spec=gs_node
    )
    for i in range(dim[0]):
        for j in range(dim[1]):
            element = 'i{}j{}'.format(i, j)
            sub_node = node.find(element)
            sub_type = sub_node.attrib['type']
            if sub_type in ['Hierarchical', 'UniformHierarchical']:
                shared_col_b[i]['children'], shared_row_b[j]['children'] = (
                    plot_hierarchical(
                        sub_node,
                        gs_subgrid[i, j],
                        (level == 0 and [i, j] in [[0, 0], [0, 1]]) or color
                    )
                )
            elif sub_type == 'LowRank':
                plot_lowrank(sub_node, gs_subgrid[i, j])
                # plot_lowrank_patches(
                #     sub_node, gs_subgrid[i, j], dim
                #     # (level == 0 and [i, j] in [[0, 0], [0, 1]]) or (color and i >= j)
                # )
            elif sub_type == 'LowRankShared':
                width = plot_lowrank_shared_patches(
                    sub_node, gs_subgrid[i, j], dim
                    # (level == 0 and [i, j] in [[0, 0], [0, 1]]) or (color and i >= j)
                )
                shared_col_b[i]['exists'] = True
                shared_col_b[j]['width'] = width
                shared_row_b[j]['exists'] = True
                shared_row_b[i]['width'] = width
            elif sub_type == 'Dense':
                plot_dense(sub_node, gs_subgrid[i, j])
                # plot_dense_patch(
                #     sub_node, gs_subgrid[i, j]
                #     # (level == 0 and [i, j] in [[0, 0], [0, 1]]) or (color and i >= j),
                #     # 'lower'
                # )
    return shared_col_b, shared_row_b


def plot_lowrank(xml_node, gs_node):
    dim = [int(xml_node.attrib['dim0']), int(xml_node.attrib['dim1'])]
    svalues = [float(x)+1e-10 for x in xml_node.attrib['svalues'].split(',')]
    slog = np.log(svalues[0:min(num_singular_values, *dim)])
    ax = plt.subplot(gs_node)

    count = 0
    for sv in svalues:
        if sv > tolerance:
            count += 1

    ax.text(0.5, 0.5, f"{count}")

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
    xml_node, gs_node, dim, has_color=True
):
    level = int(xml_node.attrib['level'])
    ax = plt.subplot(gs_node)
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


def plot_lowrank_shared_patches(
    xml_node, gs_node, dim, has_color=True
):
    level = int(xml_node.attrib['level'])
    ax = plt.subplot(gs_node)
    ax.set(xticks=[], yticks=[])
    dist = 0.0075*2**(level*np.sqrt(dim[0]/2)-int(level/4))
    k = (1.0 - 3*dist)/40*2**(level*np.sqrt(dim[0]/2)-int(level/4))
    n = (1.0 - 3*dist-k)
    color = yrpink if has_color else "grey"
    S = mpatch.Rectangle(
        (dist, 2*dist+n),
        k, k,
        edgecolor='none',
        facecolor=color
    )
    # Add the patch to the Axes
    ax.add_patch(S)
    return k


def plot_dense(xml_node, gs_node):
    dim = [int(xml_node.attrib['dim0']), int(xml_node.attrib['dim1'])]
    svalues = [float(x)+1e-10 for x in xml_node.attrib['svalues'].split(',')]
    slog = np.log(svalues[0:min(num_singular_values, *dim)])
    ax = plt.subplot(gs_node)

    count = 0
    for sv in svalues:
        if sv > tolerance:
            count += 1
    ax.text(0.5, 0.5, f"{count}")
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


def plot_dense_patch(xml_node, gs_node, has_color=True, ul=None):
    ax = plt.subplot(gs_node)
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

def plot_shared_basis(shared_basis, gs_root, transpose, width_factor):
    gs_split = gs.GridSpecFromSubplotSpec(
        1 if transpose else len(shared_basis),
        len(shared_basis) if transpose else 1,
        subplot_spec=gs_root
    )
    child_depths = [0]
    for i, basis in enumerate(shared_basis):
        gs_own, gs_children = None, None
        if not basis['exists'] and basis['children'] is None:
            continue
        if basis['exists'] and basis['children'] is None:
            gs_own = gs_split[i]
            gs_children = None
        elif not basis['exists'] and basis['children'] is not None:
            gs_own = None
            gs_children = gs_split[i]
        elif basis['exists'] and basis['children'] is not None:
            gs_own, gs_children = gs.GridSpecFromSubplotSpec(
                2 if transpose else 1,
                1 if transpose else 2,
                subplot_spec=gs_split[i]
            )
        else:
            print(basis['exists'], basis['children'])
            raise ValueError
        depth = 0
        if gs_children is not None:
            depth += plot_shared_basis(
                basis['children'], gs_children, transpose, width_factor
            )
            if gs_own is not None:
                if transpose:
                    gs_split[i].set_height_ratios(1, depth)
                else:
                    gs_split[i].set_width_ratios(1, depth)
        if gs_own is not None:
            depth += 1
            ax = plt.subplot(gs_own)
            ax.axis('off')
            level = basis['level']
            dim = basis['dim']
            width = basis['width']*width_factor/dim
            base = mpatch.Rectangle(
                (
                    0.05 if transpose else 0.5-width/2,
                    0.5-width/2 if transpose else 0.05
                ),
                0.9 if transpose else width, width if transpose else 0.9,
                edgecolor='none',
                facecolor=yrpink
            )
            ax.add_patch(base)
        child_depths.append(depth)
    return max(child_depths)


def main():
    in_file = 'matrix.xml' if len(sys.argv) == 1 else sys.argv[1]
    tree = read_xml(in_file)
    root = tree.getroot()

    fig = plt.figure(figsize=(5, 5), dpi=200)
    grid = fig.add_gridspec(2, 2)
    shared_col_b, shared_row_b = plot_hierarchical(
        root,
        grid[1, 1]
    )
    width_factor = 20
    col_basis_width = plot_shared_basis(
        shared_col_b, grid[1, 0], False, width_factor
    )
    grid.set_width_ratios([col_basis_width, width_factor])
    row_basis_height = plot_shared_basis(
        shared_row_b, grid[0, 1], True, width_factor
    )
    grid.set_height_ratios([row_basis_height, width_factor])

    # TODO change figure size if width of bases varies (.set_size_inches)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(in_file + ".pdf")
    # plt.show()


if __name__ == '__main__':
    main()
