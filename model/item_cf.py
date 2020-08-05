import logging
import numpy as np

from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode

# top k items to recommend
TOP_K = 10

if __name__ == "__main__":
    data = movielens.load_pandas_df(
        size='100k',
        header=['UserId', 'MovieId', 'Rating', 'Timestamp'],
        title_col='Title'
    )

    # Convert the float precision to 32-bit in order to reduce memory consumption
    data.loc[:, 'Rating'] = data['Rating'].astype(np.float32)

    header = {
        "col_user": "UserId",
        "col_item": "MovieId",
        "col_rating": "Rating",
        "col_timestamp": "Timestamp",
        "col_prediction": "Prediction",
    }

    train, test = python_stratified_split(data, ratio=0.75, col_user=header["col_user"], col_item=header["col_item"],
                                          seed=42)

    # set log level to INFO
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    model = SARSingleNode(
        similarity_type="jaccard",
        time_decay_coefficient=30,
        time_now=None,
        timedecay_formula=True,
        **header
    )

    model.fit(train)

    top_k = model.recommend_k_items(test, remove_seen=True)

    top_k_with_titles = (top_k.join(data[['MovieId', 'Title']].drop_duplicates().set_index('MovieId'),
                                    on='MovieId',
                                    how='inner').sort_values(by=['UserId', 'Prediction'], ascending=False))

    args = [test, top_k]
    kwargs = dict(col_user='UserId',
                  col_item='MovieId',
                  col_rating='Rating',
                  col_prediction='Prediction',
                  relevancy_method='top_k',
                  k=TOP_K)

    eval_map = map_at_k(*args, **kwargs)
    eval_ndcg = ndcg_at_k(*args, **kwargs)
    eval_precision = precision_at_k(*args, **kwargs)
    eval_recall = recall_at_k(*args, **kwargs)

    print(f"Model:",
          f"Top K:\t\t {TOP_K}",
          f"MAP:\t\t {eval_map:f}",
          f"NDCG:\t\t {eval_ndcg:f}",
          f"Precision@K:\t {eval_precision:f}",
          f"Recall@K:\t {eval_recall:f}", sep='\n')
