import numpy as np
import time
import scipy.sparse as sparse
from lightfm import LightFM
from collections import defaultdict


def main():
    sparse_training_matrix, training_tuples = read_training_data()
    ground_truth = read_ground_truth()

    start_time = time.time()

    model = LightFM(no_components=30, learning_rate=0.05, loss='bpr')
    model.fit(sparse_training_matrix, epochs=100)
    URB, URE = model.get_user_representations()
    IRB, IRE = model.get_item_representations()