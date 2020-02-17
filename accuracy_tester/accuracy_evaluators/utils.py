import itertools
import progressbar


def evaluate_outputs(activations, top_n, answer):
    assert activations.shape == (1001,) or activations.shape == (1000,)

    if activations.shape[0] == 1001:
        def index_to_label(index: int): return str(index)
    else:
        def index_to_label(index: int): return str(index + 1)

    indices = list(map(
        lambda pair: pair[0],
        itertools.islice(
            sorted(
                enumerate(activations),
                key=lambda pair: -pair[1]
            ),
            top_n
        )
    ))
    ret = []
    current = False
    for index in indices:
        current = current or (index_to_label(index) == answer)
        ret.append(int(current))
    return ret


def count_dataset_size(image_path_label_gen):
    """count_dataset_size

    Returns:
        (generator, int)
    """
    image_path_label_gen, gen = itertools.tee(image_path_label_gen)
    ret = 0
    for _ in gen:
        ret += 1
    return (image_path_label_gen, ret)


def construct_evaluating_progressbar(dataset_size, model_basename):
    return progressbar.ProgressBar(max_value=dataset_size, widgets=[
        'Evaluating {} ['.format(model_basename),
        progressbar.Timer(),
        '] ',
        progressbar.Bar(),
        ' (',
        progressbar.ETA(),
        ') ',
    ])
