import xml.etree.ElementTree as ET
from FRANK.dense import Dense

def printf(s, n=None):
    if n is None:
        s = '--- ' + s + ' '
        print("{:-<37}".format(s))
    else:
        print('{:25} : {:3.1e}'.format(s, n))


def print_time(s, n):
    print('{:25} : {:0<9f}'.format(s, n))



def fill_xml(A, node):
    node.set('type', A.type())
    node.set('dim0', str(A.dim[0]))
    node.set('dim1', str(A.dim[1]))
    node.set('level', str(A.level))
    if A.type() == 'Dense':
        S = Dense(None, None, min(A.dim), min(A.dim))
        Dense(A).svd_s_only(S)
        singular_values = str(S[0, 0])
        for i in range(1, min(A.dim)):
            singular_values += "," + str(S[i, i])
        node.set('svalues', singular_values)
    elif A.type() == 'LowRank':
        S = Dense(None, None, min(A.dim), min(A.dim))
        Dense(A).svd_s_only(S)
        singular_values = str(S[0, 0])
        for i in range(1, min(A.dim)):
            singular_values += "," + str(S[i, i])
        node.set('svalues', singular_values)
    elif A.type() == 'Hierarchical':
        for i in range(A.dim[0]):
            for j in range(A.dim[1]):
                subel = ET.SubElement(node, 'i{}j{}'.format(i, j))
                fill_xml(A[i, j], subel)
        pass
    else:
        raise(ValueError)


def print_xml(A):
    tree = ET.ElementTree(ET.Element('root'))
    root = tree.getroot()
    fill_xml(A, root)
    tree.write(open('matrix.xml', 'w'), encoding='unicode')
