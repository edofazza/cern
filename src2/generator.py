from torch import nn


class Generator(nn.Module):
    def __init__(self, input_size, output_size, num_classifiers, transformer_hidden_size=256, num_transformer_layers=2):
        super(Generator, self).__init__()
        self.output_size = output_size
        self.num_classifier = num_classifiers
        self.fc = nn.Linear(input_size, transformer_hidden_size)
        self.transformer = nn.Transformer(
            d_model=transformer_hidden_size,
            nhead=4,  # You can adjust the number of attention heads
            num_encoder_layers=num_transformer_layers,
            num_decoder_layers=num_transformer_layers
        )
        self.fc_output = nn.Linear(transformer_hidden_size, output_size * num_classifiers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(1, -1, x.size(1))  # Add batch dimension
        x = self.transformer(x, x)
        x = x.mean(dim=1)  # Average over the sequence dimension
        x = self.fc_output(x)
        return x.view(-1, self.output_size, self.num_classifiers)