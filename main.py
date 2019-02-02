import scipy.sparse as sparse
from lightfm import LightFM


def main():
    # sparse_training_matrix is user-item interaction matrix.

    model = LightFM(no_components=30, learning_rate=0.05, loss='bpr', item_pretrain=True, item_pretrain_file='item_embeddings.txt')
    model.fit(sparse_training_matrix, epochs=100)
    URB, URE = model.get_user_representations()
    IRB, IRE = model.get_item_representations()
