# import numpy as np
import torch
from torch_geometric.data import download_url, extract_zip
from torch_sparse import SparseTensor


class DatasetUtils:
    # load edges between users and movies
    def load_edge_csv(
        df, src_index_col, dst_index_col, link_index_col, rating_threshold=3.5
    ):
        """Loads csv containing edges between users and items

        Args:
            path (str): path to csv file
            src_index_col (str): column name of users
            src_mapping (dict): mapping between row number and user id
            dst_index_col (str): column name of items
            dst_mapping (dict): mapping between row number and item id
            link_index_col (str): column name of user item interaction
            rating_threshold (int, optional): Threshold to determine positivity of edge. Defaults to 4.

        Returns:
            torch.Tensor: 2 by N matrix containing the node ids of N user-item edges
        """

        edge_index = None
        src = [user_id for user_id in df["userId"]]

        num_users = len(df["userId"].unique())

        dst = [(movie_id) for movie_id in df["movieId"]]

        link_vals = df[link_index_col].values

        edge_attr = (
            torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long)
            >= rating_threshold
        )

        edge_values = []

        edge_index = [[], []]
        for i in range(edge_attr.shape[0]):
            if edge_attr[i]:
                edge_index[0].append(src[i])
                edge_index[1].append(dst[i])
                edge_values.append(link_vals[i])

        # edge_values is the label we will use for compute training loss
        return edge_index, edge_values

    def convert_r_mat_edge_index_to_adj_mat_edge_index(
        input_edge_index, input_edge_values, num_users, num_movies
    ):
        R = torch.zeros((num_users, num_movies))
        for i in range(len(input_edge_index[0])):
            row_idx = input_edge_index[0][i]
            col_idx = input_edge_index[1][i]
            R[row_idx][col_idx] = input_edge_values[
                i
            ]  # assign actual edge value to Interaction Matrix

        R_transpose = torch.transpose(R, 0, 1)

        # create adj_matrix
        adj_mat = torch.zeros((num_users + num_movies, num_users + num_movies))
        adj_mat[:num_users, num_users:] = R.clone()
        adj_mat[num_users:, :num_users] = R_transpose.clone()

        adj_mat_coo = adj_mat.to_sparse_coo()
        adj_mat_coo_indices = adj_mat_coo.indices()
        adj_mat_coo_values = adj_mat_coo.values()
        return adj_mat_coo_indices, adj_mat_coo_values

    def convert_adj_mat_edge_index_to_r_mat_edge_index(
        input_edge_index, input_edge_values, num_users, num_movies
    ):
        sparse_input_edge_index = SparseTensor(
            row=input_edge_index[0],
            col=input_edge_index[1],
            value=input_edge_values,
            sparse_sizes=((num_users + num_movies), num_users + num_movies),
        )

        adj_mat = sparse_input_edge_index.to_dense()
        interact_mat = adj_mat[:num_users, num_users:]

        r_mat_edge_index = interact_mat.to_sparse_coo().indices()
        r_mat_edge_values = interact_mat.to_sparse_coo().values()

        return r_mat_edge_index, r_mat_edge_values

    def load_dataset():
        # download the dataset
        # https://grouplens.org/datasets/movielens/
        # "Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018"
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        extract_zip(download_url(url, "."), ".")

        movie_path = "./ml-latest-small/movies.csv"
        rating_path = "./ml-latest-small/ratings.csv"
        user_path = "./ml-latest-small/users.csv"

        return movie_path, rating_path, user_path
