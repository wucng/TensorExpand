[Various Embeddings](https://github.com/tensorpack/tensorpack/tree/master/examples/SimilarityLearning)

在MNIST上重现一些嵌入方法：

```python
# to train:
./mnist-embeddings.py --algorithm [siamese/cosine/triplet/softtriplet/center]
# to visualize:
./mnist-embeddings.py --algorithm [siamese/cosine/triplet/softtriplet/center] --visualize --load train_log/mnist-embeddings/checkpoint
```

![这里写图片描述](https://github.com/tensorpack/tensorpack/raw/master/examples/SimilarityLearning/results.jpg)