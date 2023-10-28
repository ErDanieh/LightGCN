import torch
from torch import nn, optim, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from collections import defaultdict

from utils.DatasetUtils import DatasetUtils


# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126"""

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        K=3,
        add_self_loops=False,
        dropout_rate=0.1,
    ):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops

        # define user and item embedding for direct look up.
        # embedding dimension: num_user/num_item x embedding_dim
        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim
        )  # e_u^0

        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim
        )  # e_i^0

        # "Fills the input Tensor with values drawn from the normal distribution"
        # according to LightGCN paper, this gives better performance
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        # create a linear layer (fully connected layer) so we can output a single value (predicted_rating)
        self.out = nn.Linear(embedding_dim + embedding_dim, 1)

    def forward(self, edge_index: Tensor, edge_values: Tensor, num_users, num_movies):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """

        """
            compute \tilde{A}: symmetrically normalized adjacency matrix
            \tilde_A = D^(-1/2) * A * D^(-1/2)    according to LightGCN paper
        
            this is essentially a metrix operation way to get 1/ (sqrt(n_neighbors_i) * sqrt(n_neighbors_j))

        
            if your original edge_index look like
            tensor([[   0,    0,    0,  ...,  609,  609,  609],
                    [   0,    2,    5,  ..., 9444, 9445, 9485]])
                    
                    torch.Size([2, 99466])
                    
            then this will output: 
                (
                 tensor([[   0,    0,    0,  ...,  609,  609,  609],
                         [   0,    2,    5,  ..., 9444, 9445, 9485]]), 
                 tensor([0.0047, 0.0096, 0.0068,  ..., 0.0592, 0.0459, 0.1325])
                 )
                 
              where edge_index_norm[0] is just the original edge_index
              
              and edge_index_norm[1] is the symmetrically normalization term. 
              
            under the hood it's basically doing
                def compute_gcn_norm(edge_index, emb):
                    emb = emb.weight
                    from_, to_ = edge_index
                    deg = degree(to_, emb.size(0), dtype=emb.dtype)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                    norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

                    return norm
                 
                
        """
        edge_index_norm = gcn_norm(
            edge_index=edge_index, add_self_loops=self.add_self_loops
        )

        # concat the user_emb and item_emb as the layer0 embing matrix
        # size will be (n_users + n_items) x emb_vector_len.   e.g: 10334 x 64
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])  # E^0

        embs = [emb_0]  # save the layer0 emb to the embs list

        # emb_k is the emb that we are actually going to push it through the graph layers
        # as described in lightGCN paper formula 7
        emb_k = emb_0

        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for i in range(self.K):
            emb_k = self.propagate(
                edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1]
            )
            embs.append(emb_k)

        # this is doing the formula8 in LightGCN paper

        # the stacked embs is a list of embedding matrix at each layer
        #    it's of shape n_nodes x (n_layers + 1) x emb_vector_len.
        #        e.g: torch.Size([10334, 4, 64])
        embs = torch.stack(embs, dim=1)

        # From LightGCn paper: "In our experiments, we find that setting α_k uniformly as 1/(K + 1)
        #    leads to good performance in general."
        emb_final = torch.mean(embs, dim=1)  # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]
        )  # splits into e_u^K and e_i^K

        (
            r_mat_edge_index,
            _,
        ) = DatasetUtils.convert_adj_mat_edge_index_to_r_mat_edge_index(
            edge_index, edge_values, num_users, num_movies
        )

        src, dest = r_mat_edge_index[0], r_mat_edge_index[1]

        # applying embedding lookup to get embeddings for src nodes and dest nodes in the edge list
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]

        # output dim: edge_index_len x 128 (given 64 is the original emb_vector_len)
        output = torch.cat([user_embeds, item_embeds], dim=1)

        # push it through the linear layer
        output = self.out(output)

        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def get_recall_at_k(
        input_edge_index,
        input_edge_values,  # the true label of actual ratings for each user/item interaction
        pred_ratings,  # the list of predicted ratings
        k=10,
        threshold=3.5,
    ):
        with torch.no_grad():
            user_item_rating_list = defaultdict(list)

            for i in range(len(input_edge_index[0])):
                src = input_edge_index[0][i].item()
                dest = input_edge_index[1][i].item()
                true_rating = input_edge_values[i].item()
                pred_rating = pred_ratings[i].item()

                user_item_rating_list[src].append((pred_rating, true_rating))

            recalls = dict()
            precisions = dict()

            for user_id, user_ratings in user_item_rating_list.items():
                user_ratings.sort(key=lambda x: x[0], reverse=True)

                n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

                n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

                n_rel_and_rec_k = sum(
                    ((true_r >= threshold) and (est >= threshold))
                    for (est, true_r) in user_ratings[:k]
                )

                precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
                recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

            overall_recall = sum(rec for rec in recalls.values()) / len(recalls)
            overall_precision = sum(prec for prec in precisions.values()) / len(
                precisions
            )

            return overall_recall, overall_precision
