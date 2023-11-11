import torch.nn
import torch
import torchtext.datasets
import torchtext.data.utils
import torch.utils.data.dataset
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(torch.nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def data_process(raw_text) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# def train(model: torch.nn.Module) -> None:
#     model.train()  # turn on train mode
#     total_loss = 0.
#     log_interval = 200
#     start_time = time.time()
#     src_mask = generate_square_subsequent_mask(bptt).to(device)
#
#     num_batches = len(train_data) // bptt
#     for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
#         data, targets = get_batch(train_data, i)
#         seq_len = data.size(0)
#         if seq_len != bptt:  # only on last batch
#             src_mask = src_mask[:seq_len, :seq_len]
#         output = model(data, src_mask)
#         loss = criterion(output.view(-1, ntokens), targets)
#
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#         optimizer.step()
#
#         total_loss += loss.item()
#         if batch % log_interval == 0 and batch > 0:
#             lr = scheduler.get_last_lr()[0]
#             ms_per_batch = (time.time() - start_time) * 1000 / log_interval
#             cur_loss = total_loss / log_interval
#             ppl = math.exp(cur_loss)
#             print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
#                   f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
#                   f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
#             total_loss = 0
#             start_time = time.time()


def evaluate(model: torch.nn.Module) -> torch.Tensor:
    to_gen = 30

    model.eval()  # turn on evaluation mode
    src_mask = generate_square_subsequent_mask(to_gen).to(device)
    sequence = torch.randint(0, ntokens, (to_gen, 1), dtype=torch.long)
    with torch.no_grad():
        output = model(sequence, src_mask)
        output_flat = output.view(-1, ntokens)
        return torch.argmax(output_flat, dim=1)


if __name__ == '__main__':
    train_iter = torchtext.datasets.WikiText2(split='train')
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 20
    eval_batch_size = 10

    ntokens = len(vocab)  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model_params_path = 'trained_transformer_wikitext2.pth'
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(model_params_path))

    print(evaluate(model))
