# Packages and modules
import torch
import torch.nn as nn
import pytorch_lightning as pl


class NNModel(pl.LightningModule):

    def __init__(self,
                 emb_dims,
                 no_of_cont,
                 lin_layer_sizes,
                 output_size,
                 emb_dropout,
                 lin_layer_dropouts,
                 loss_function,
                 learning_rate):
        """
        Parameters
        ----------
        emb_dims: list of two elements tuples
            The list contains a two element tuple for each categorical feature. The first element tuple denotes
            the number of unique values of the categorical feture. The second element tuple denotes the embedding
            dimension to be used for the feature.
        no_of_cont: int
            Number of continuous features in the data.
        lin_layer_sizes: list of int
            The list contains the size of each linear layer. The length of the list is the total number of linear
            layers in the NN.
        output_size: int
            The size of the final output.
        emb_dropout: float
            The dropout to be used after the embedding layers.
        lin_layer_dropouts: list of floats
            The dropouts to be used after each linear layer.
        loss_function: function
            Loss function to be used for the train, validation and test step.
        learning_rate: float
            Learning rate for the optimizer.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        # Number of embeddings
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList(
            [first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]
                                           ) for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])

        self.relu = nn.ReLU()

        # Loss function
        self.loss_function = loss_function

        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, batch):

        # Embeds categorical data
        if self.no_of_embs != 0:
            x = [emb_layer(batch["cat_data"][:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        # Embeds continuous data
        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(batch["cont_data"])

        # Concatenation of categorical and continuous data after initialization
        if self.no_of_embs != 0:
            x = torch.cat([x, normalized_cont_data], 1)
        else:
            x = normalized_cont_data

        # Hidden layers
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = self.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        # Output layer
        x = self.output_layer(x)

        return x

    def training_step(self, batch, idx):
        logits = self.forward(batch)
        loss = self.loss_function(logits, batch["target"])
        self.log('train_loss', loss, sync_dist=False)
        return loss

    def validation_step(self, batch, idx):
        logits = self.forward(batch)
        loss = self.loss_function(logits, batch["target"])
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, idx):
        logits = self.forward(batch)
        test_loss = self.loss_function(logits, batch['target'])
        print(logits, batch["target"], test_loss)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
