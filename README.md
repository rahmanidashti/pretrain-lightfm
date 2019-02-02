# Pre-train LightFM
Pre-train embedding in LightFM recommender system framework

## How to use
When you install LightFM package, you will replace the `lightfm.py` with the original one. Here, we just implement the `Item Embedding` but you can follow the structure and implement the `User Embedding`. You can use a `.txt` file that each row shows a user of item embedding and columns are the embeddings.

## Method

```Python
def __init__(self, no_components=10, k=5, n=10,
                 learning_schedule='adagrad',
                 loss='logistic',
                 learning_rate=0.05, rho=0.95, epsilon=1e-6,
                 item_alpha=0.0, user_alpha=0.0, max_sampled=10,
                 random_state=None, user_pretrain= False, user_pretrain_file=None,
                 item_pretrain=False, item_pretrain_file=None)
```

If you set the `item_pretrain = True` then the pre-train item embedding will be considered as follows:

```Python
# Pre-train item embedding
if self.item_pretrain:
  print("Pre-Train Item Embedding Lunch.")

  Item_Embeddings_File = self.item_pretrain_file
  Item_Embeddings = open(Item_Embeddings_File, 'r').readlines()
  item_embeddings = np.ndarray((no_item_features, no_components)).astype(np.float32)

  Item = 0
  for eachline in Item_Embeddings:
    ItemVectorElements = eachline.split()
    for element in range(0, no_components):
      item_embeddings[Item][element] = ItemVectorElements[element]
    Item = Item + 1

  self.item_embeddings = poi_embeddings

  print("Pre-Train Item Embedding Finished.")

else:
  self.item_embeddings = ((self.random_state.rand(no_item_features, no_components) - 0.5) / no_components).astype(np.float32)
```
