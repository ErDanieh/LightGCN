
from model.LightGCN import LightGCN
from utils.DatasetUtils import DatasetUtils

from sklearn import preprocessing
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
import matplotlib.pyplot as plt


def main():
    movie_path, rating_path, user_path = DatasetUtils.load_dataset()

    # Prepare data for LightGCN model

    rating_df = pd.read_csv(rating_path)

    lbl_user = preprocessing.LabelEncoder()
    lbl_movie = preprocessing.LabelEncoder()

    rating_df.userId = lbl_user.fit_transform(rating_df.userId.values)
    rating_df.movieId = lbl_movie.fit_transform(rating_df.movieId.values)

    edge_index, edge_values = DatasetUtils.load_edge_csv(
        rating_df,
        src_index_col='userId',
        dst_index_col='movieId',
        link_index_col='rating',
        rating_threshold=1 # need to use threshold=1 so the model can learn based on RMSE
    )

    edge_index = torch.LongTensor(edge_index) 
    edge_values = torch.tensor(edge_values)

    num_users = len(rating_df['userId'].unique())
    num_movies = len(rating_df['movieId'].unique())

    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(
        all_indices, 
        test_size=0.2, 
        random_state=1
    )

    val_indices, test_indices = train_test_split(
        test_indices, 
        test_size=0.5, 
        random_state=1
    )

    train_edge_index = edge_index[:, train_indices]
    train_edge_value = edge_values[train_indices]

    val_edge_index = edge_index[:, val_indices]
    val_edge_value = edge_values[val_indices]

    test_edge_index = edge_index[:, test_indices]
    test_edge_value = edge_values[test_indices]

    train_edge_index, train_edge_values  = DatasetUtils.convert_r_mat_edge_index_to_adj_mat_edge_index(train_edge_index, train_edge_value, num_users, num_movies)
    val_edge_index, val_edge_values = DatasetUtils.convert_r_mat_edge_index_to_adj_mat_edge_index(val_edge_index, val_edge_value, num_users, num_movies)
    test_edge_index, test_edge_values = DatasetUtils.convert_r_mat_edge_index_to_adj_mat_edge_index(test_edge_index, test_edge_value, num_users, num_movies)

    layers = 1 
    model = LightGCN(num_users=num_users, 
                    num_items=num_movies, 
                    K=layers)

    # define contants
    ITERATIONS = 1
    EPOCHS = 10

    BATCH_SIZE = 1024

    LR = 1e-3
    ITERS_PER_EVAL = 200
    ITERS_PER_LR_DECAY = 200
    K = 10
    LAMBDA = 1e-6

    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}.")


    model = model.to(device)
    model.train()

    # add decay to avoid overfit
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)


    loss_func = nn.MSELoss()
    
    r_mat_train_edge_index, r_mat_train_edge_values = DatasetUtils.convert_adj_mat_edge_index_to_r_mat_edge_index(train_edge_index, train_edge_values, num_users, num_movies)
    r_mat_val_edge_index, r_mat_val_edge_values = DatasetUtils.convert_adj_mat_edge_index_to_r_mat_edge_index(val_edge_index, val_edge_values, num_users, num_movies)
    r_mat_test_edge_index, r_mat_test_edge_values = DatasetUtils.convert_adj_mat_edge_index_to_r_mat_edge_index(test_edge_index, test_edge_values, num_users, num_movies)

    
    # training loop
    train_losses = []
    val_losses = []
    val_recall_at_ks = []

    for iter in range(ITERATIONS):
        # forward propagation

        # the rating here is based on r_mat
        pred_ratings = model.forward(train_edge_index, train_edge_values, num_users, num_movies)
        

        train_loss = loss_func(pred_ratings, r_mat_train_edge_values.view(-1,1))    

            
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # going over validation set
        if iter % ITERS_PER_EVAL == 0:
            model.eval()

            with torch.no_grad():
                val_pred_ratings = model.forward(val_edge_index, val_edge_values, num_users, num_movies)
                
                val_loss = loss_func(val_pred_ratings, r_mat_val_edge_values.view(-1,1)).sum()
                
                recall_at_k, precision_at_k = LightGCN.get_recall_at_k(r_mat_val_edge_index, 
                                                            r_mat_val_edge_values, 
                                                            val_pred_ratings, 
                                                            k = 20
                                                            )
        
                    
                val_recall_at_ks.append(round(recall_at_k, 5))
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
            
                print(f"[Iteration {iter}/{ITERATIONS}], train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss.item(), 5)},  recall_at_k {round(recall_at_k, 5)}, precision_at_k {round(precision_at_k, 5)}")

            model.train()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()
            
    iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, val_losses, label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training and validation loss curves')
    plt.legend()
    plt.show()
    
    f2 = plt.figure()
    plt.plot(iters, val_recall_at_ks, label='recall_at_k')
    plt.xlabel('iteration')
    plt.ylabel('recall_at_k')
    plt.title('recall_at_k curves')
    plt.show()
    
    model.eval()
    with torch.no_grad():
        pred_ratings = model.forward(test_edge_index, test_edge_values, num_users, num_movies)
        recall_at_k, precision_at_k = LightGCN.get_recall_at_k(r_mat_test_edge_index, 
                                                    r_mat_test_edge_values, 
                                                    pred_ratings, 20)
        print(f"recall_at_k {round(recall_at_k, 5)}, precision_at_k {round(precision_at_k, 5)}")


if __name__ == "__main__":
    main()