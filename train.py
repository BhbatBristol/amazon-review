from data import get_graph
import torch
from model import HeteroGraphSAGE
import torch.utils.data as D
from dgl.dataloading import NeighborSampler
from dgl.dataloading import NodeDataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm
import torchmetrics as tm
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(num_epochs):
    graph = get_graph()

    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1] for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], dims[rel[2]]) for rel in graph.canonical_etypes}

    model = HeteroGraphSAGE(input_dropout=0.2,
                                dropout=0.2,
                                hidden_dim=512,
                                feat_dict=feat_dict)
    model.to(device)
    model.train()

    node_enum = torch.arange(graph.num_nodes('review'))
    torch_gen = torch.Generator().manual_seed(4242)

    num_train = int(0.8 * graph.num_nodes('review'))
    num_val = int(0.1 * graph.num_nodes('review'))
    num_test = graph.num_nodes('review') - (num_train + num_val)
    nums = [num_train, num_val, num_test]


    train_nids, val_nids, test_nids = D.random_split(dataset=node_enum.long(), lengths=nums, generator=torch_gen)

    train_nids = {'review': train_nids}
    val_nids = {'review': val_nids}
    test_nids = {'review': test_nids}

    sampler = NeighborSampler([100], replace=False)

    train_dataloader = NodeDataLoader(graph=graph,
                                      indices=train_nids,
                                      graph_sampler=sampler,
                                      batch_size=32,
                                      shuffle=True,
                                      drop_last=False)
    val_dataloader = NodeDataLoader(graph=graph,
                                    indices=val_nids,
                                    graph_sampler=sampler,
                                    batch_size=1000000,
                                    shuffle=False,
                                    drop_last=False)
    test_dataloader = NodeDataLoader(graph=graph,
                                     indices=test_nids,
                                     graph_sampler=sampler,
                                     batch_size=1000000,
                                     shuffle=False,
                                     drop_last=False)

    # pos_weight_tensor = torch.tensor(20.).to(device)
    class_weights = [1., 1.]
    pos_weight_tensor = torch.tensor(class_weights).to(device)


    opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    scheduler = LinearLR(optimizer=opt, start_factor=1., end_factor=1e-7 / 3e-4, total_iters=100)
    scorer = tm.classification.F1Score(task="multiclass", num_classes=2, average='none').to(device)
    epoch_pbar = tqdm(range(num_epochs), desc='Training')

    # print(type(graph.nodes['review'].data['feat']))
    # exit()

    for epoch in epoch_pbar:
        train_loss = 0.0
        val_loss = 0.0

        scorer.reset()

        model.train()
        for _, _, blocks in train_dataloader:
            opt.zero_grad()
            blocks = [block.to(device) for block in blocks]


            input_feats = {n: feat.float() for n, feat in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label']['review'].to(device)


            # Forward propagation
            logits = model(blocks, input_feats).squeeze()


            # Compute loss
            loss = F.nll_loss(input=F.log_softmax(logits, dim=-1),
                              target=output_labels.long(),
                              weight=pos_weight_tensor)

            scorer(logits.argmax(axis=-1), output_labels)

            loss.backward()
            opt.step()

            train_loss += float(loss)

        train_loss /= len(train_dataloader)

        train_f1s = scorer.compute()
        train_misinformation_f1 = train_f1s[0].item()
        train_factual_f1 = train_f1s[1].item()

        scorer.reset()

        model.eval()
        for _, _, blocks in val_dataloader:
            with torch.no_grad():
                blocks = [block.to(device) for block in blocks]

                input_feats = {n: f.float() for n, f in blocks[0].srcdata['feat'].items()}
                output_labels = blocks[-1].dstdata['label']['review'].to(device)


                logits = model(blocks, input_feats).squeeze()

                loss = F.nll_loss(input=F.log_softmax(logits, dim=-1),
                                  target=output_labels.long(),
                                  weight=pos_weight_tensor)

                scorer(logits.argmax(axis=-1), output_labels)
                val_loss += float(loss)

        val_loss /= len(val_dataloader)

        val_f1s = scorer.compute()
        val_misinformation_f1 = val_f1s[0].item()
        val_factual_f1 = val_f1s[1].item()

        # Update progress bar description
        desc = (f'Training - '
                f'loss {train_loss:.3f} - '
                f'pos_f1 {train_factual_f1:.3f} - '
                f'neg_f1 {train_misinformation_f1:.3f} - '
                f'val_loss {val_loss:.3f} - '
                f'val_factual_f1 {val_factual_f1:.3f} - '
                f'val_neg_f1  {val_misinformation_f1:.3f}')
        epoch_pbar.set_description(desc)
        scheduler.step()

        # Close progress bar
    epoch_pbar.close()
    val_loss = 0.0
    test_loss = 0.0

    # Reset metrics
    scorer.reset()

    # Final evaluation on the validation set
    model.eval()
    for _, _, blocks in tqdm(val_dataloader, desc='Evaluating'):
        with torch.no_grad():
            # Ensure that `blocks` are on the correct device
            blocks = [block.to(device) for block in blocks]

            # Get the input features and the output labels
            input_feats = {n: f.float()
                           for n, f in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label']['review'].to(device)


            # Forward propagation
            logits = model(blocks, input_feats).squeeze()

            # Compute validation loss
            loss = F.nll_loss(
                input=F.log_softmax(logits, dim=-1),
                target=output_labels.long(),
                weight=pos_weight_tensor
            )

            # Compute validation metrics
            scorer(logits.argmax(axis=-1), output_labels)

            # Store the validation loss
            val_loss += float(loss)

    # Divide the validation loss by the number of batches
    val_loss /= len(val_dataloader)

    # Compute the validation metrics
    val_f1s = scorer.compute()
    val_misinformation_f1 = val_f1s[0].item()
    val_factual_f1 = val_f1s[1].item()

    # Reset the metrics
    scorer.reset()

    # Final evaluation on the test set
    model.eval()
    for _, _, blocks in tqdm(test_dataloader, desc='Evaluating'):
        with torch.no_grad():
            blocks = [block.to(device) for block in blocks]

            input_feats = {n: f.float()
                           for n, f in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label']['review'].to(device)


            logits = model(blocks, input_feats).squeeze()

            loss = F.nll_loss(
                input=F.log_softmax(logits, dim=-1),
                target=output_labels.long(),
                weight=pos_weight_tensor
            )

            scorer(logits.argmax(axis=-1), output_labels)

            test_loss += float(loss)

    test_loss /= len(test_dataloader)

    test_f1s = scorer.compute()
    test_misinformation_f1 = test_f1s[0].item()
    test_factual_f1 = test_f1s[1].item()

    results = {
        'train': {
            'loss': train_loss,
            'pos_f1': train_factual_f1,
            'neg_f1 ': train_misinformation_f1
        },
        'val': {
            'loss': val_loss,
            'pos_f1': val_factual_f1,
            'neg_f1 ': val_misinformation_f1
        },
        'test': {
            'loss': test_loss,
            'pos_f1': test_factual_f1,
            'neg_f1 ': test_misinformation_f1
        }
    }
    print(results)
    return results

results = train(num_epochs=100)