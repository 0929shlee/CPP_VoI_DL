import gnn
import cnn
import dnn
from mininet_dataset import CppDataset
import shfile
import numpy as np
import matplotlib.pyplot as plt


def _get_accuracy(analysis_result_1d, predict_aoi_1d):
    idx_aoi_1d = [(idx_graph, predict_aoi) for idx_graph, predict_aoi in enumerate(predict_aoi_1d)]
    idx_aoi_sorted_1d = sorted(idx_aoi_1d, key=lambda item: item[1])
    predict_result_1d = [analysis_result_1d[idx] for idx, _ in idx_aoi_sorted_1d]

    analysis_result_sorted_1d = sorted(analysis_result_1d)
    accuracy_1d = [1 - abs(analysis_result_sorted_1d[idx] - predict_result_1d[idx]) / analysis_result_sorted_1d[idx] for idx in range(len(predict_result_1d))]
    accuracy = sum(accuracy_1d) / len(accuracy_1d)
    return accuracy


def _write_predict_aoi_1d(name_method, predict_aoi_1d):
    idx_aoi_1d = [(idx_graph, predict_aoi) for idx_graph, predict_aoi in enumerate(predict_aoi_1d)]
    idx_aoi_sorted_1d = sorted(idx_aoi_1d, key=lambda item: item[1])
    str_idx_aoi = '\n'.join(['{}\t{}'.format(item[0], item[1]) for item in idx_aoi_1d])
    str_idx_aoi_sorted = '\n'.join(['{}\t{}'.format(item[0], item[1]) for item in idx_aoi_sorted_1d])
    shfile.write_file('result_' + name_method + '/predict_aoi.txt', str_idx_aoi)
    shfile.write_file('result_' + name_method + '/predict_aoi_sorted.txt', str_idx_aoi_sorted)


def _write_accuracy_1d(name_method, accuracy_1d):
    str_accuracy = '\n'.join(['{}\t{}'.format(idx, accuracy) for idx, accuracy in enumerate(accuracy_1d)])
    shfile.write_file('result_' + name_method + '/accuracy.txt', str_accuracy)


def _write_plot(name_method, accuracy_1d):
    x = np.arange(0, len(accuracy_1d), 1)
    plt.plot(x, accuracy_1d, 'g', label='accuracy')
    plt.title(name_method)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('result_' + name_method + '/dnn_accuracy.png')
    plt.show()


if __name__ == "__main__":
    cpp_dataset = CppDataset(101)
    n_epochs = 1000
    analysis_result_1d = shfile.read_analysis_result_1d()

    predict_aoi_dnn_2d = dnn.run(cpp_dataset, n_epochs)
    accuracy_1d = [_get_accuracy(analysis_result_1d, predict_aoi_dnn_1d) for predict_aoi_dnn_1d in predict_aoi_dnn_2d]
    idx_accuracy_sorted_1d = sorted(enumerate(accuracy_1d), key=lambda item: item[1], reverse=True)
    best_epoch = idx_accuracy_sorted_1d[0][0]
    best_predict_aoi_dnn_1d = predict_aoi_dnn_2d[best_epoch]
    print('\nbest_epoch: {}'.format(best_epoch))
    _write_predict_aoi_1d('dnn', best_predict_aoi_dnn_1d)
    _write_accuracy_1d('dnn', accuracy_1d)
    _write_plot('dnn', accuracy_1d)

    # predict_aoi_cnn_1d = cnn.run(cpp_dataset)
    # _write_predict_aoi('cnn', predict_aoi_cnn_1d)

    # predict_aoi_gnn_1d = gnn.run(cpp_dataset)
    # _write_predict_aoi('gnn', predict_aoi_gnn_1d)


