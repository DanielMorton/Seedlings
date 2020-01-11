# Seedlings
Code to run training for the the Kaggle Plant Seendlings Classification Competition

I recommend Python 3, specifically Anaconda. To work with my AMD GPU I use Keras with the PlaidML backend.

To train a model on an InceptionV3 neural net run

```bash
python run_inception -lr {learning_rate}
```

Learning rate is the only required parameter but others are recommened. A full list is below

```
-bn|--batch-normalization: Applies batch normalization to the input layer
-b|--batch-size:  Mini-batch size for training. Defaults to 32.
-d|--dropout: The dropout percentage used. Defaults to none.
-e|--epoch: Maximum number of epochs to train. Defaults to 10.
-lr|--learnin-rate: Initial learning rate for training.
-r|--reg: L2 regularization for the last layer. Defaults to none. 
```

So far, the best results have come from L2 normalization with size 1e-2,
a learning rate of 1e-4, the standard dropout of 0.5, batch normalization
on the input layer and the default batch size. Validation accuracy is above 98%.