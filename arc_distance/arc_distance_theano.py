# Authors: Frederic Bastien
# License: MIT
import theano
import theano.tensor as tensor


def arc_distance_theano_alloc_prepare(dtype='float64'):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    a = tensor.matrix(dtype=str(dtype))
    b = tensor.matrix(dtype=str(dtype))
    # Theano don't implement all case of tile, so we do the equivalent with alloc.
    #theta_1 = tensor.tile(a[:, 0], (b.shape[0], 1)).T
    theta_1 = tensor.alloc(a[:, 0], b.shape[0], b.shape[0]).T
    phi_1 = tensor.alloc(a[:, 1], b.shape[0], b.shape[0]).T

    theta_2 = tensor.alloc(b[:, 0], a.shape[0], a.shape[0])
    phi_2 = tensor.alloc(b[:, 1], a.shape[0], a.shape[0])

    temp = (tensor.sin((theta_2 - theta_1) / 2)**2
            +
            tensor.cos(theta_1) * tensor.cos(theta_2)
            * tensor.sin((phi_2 - phi_1) / 2)**2)
    distance_matrix = 2 * (tensor.arctan2(tensor.sqrt(temp),
                                          tensor.sqrt(1 - temp)))
    name = "arc_distance_theano_alloc"
    rval = theano.function([a, b],
                           distance_matrix,
                           name=name)
    rval.__name__ = name

    return rval


def arc_distance_theano_broadcast_prepare(dtype='float64'):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    a = tensor.matrix(dtype=str(dtype))
    b = tensor.matrix(dtype=str(dtype))

    theta_1 = a[:, 0][None, :]
    theta_2 = b[:, 0][None, :]
    phi_1 = a[:, 1][:, None]
    phi_2 = b[:, 1][None, :]

    temp = (tensor.sin((theta_2 - theta_1) / 2)**2
            +
            tensor.cos(theta_1) * tensor.cos(theta_2)
            * tensor.sin((phi_2 - phi_1) / 2)**2)
    distance_matrix = 2 * (tensor.arctan2(tensor.sqrt(temp),
                                          tensor.sqrt(1 - temp)))
    name = "arc_distance_theano_broadcast"
    rval = theano.function([a, b],
                           distance_matrix,
                           name=name)
    rval.__name__ = name

    return rval


benchmarks = (
    arc_distance_theano_alloc_prepare('float64'),
    arc_distance_theano_broadcast_prepare('float64'),
)
