import pandas as pd
from matplotlib import pyplot as plt

# Data Analyst
df_ratings = pd.read_csv('ml-latest-small/ratings.csv')
df_movies = pd.read_csv('ml-latest-small/movies.csv')
merged = pd.merge(df_ratings, df_movies, on='movieId', how='left')
print(merged)
fig = plt.figure()
ax = df_ratings.rating.value_counts(True).sort_index().plot.bar(figsize=(8,6))
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Share of Ratings', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
fig.savefig('Ratings_distribution.png')
R = pd.pivot_table(data=merged, index='userId', columns='title', values='rating')

# Model Architect
class IGMC(torch.nn.Module):
    def __init__(self):
        super(IGMC, self).__init__()
        self.rel_graph_convs = torch.nn.ModuleList()
        self.rel_graph_convs.append(RGCNConv(in_channels=4, out_channels=32,\
                                             num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32\
                                             , num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32,\
                                             num_relations=5, num_bases=4))
        self.rel_graph_convs.append(RGCNConv(in_channels=32, out_channels=32,\
                                             num_relations=5, num_bases=4))
        self.linear_layer1 = Linear(256, 128)
        self.linear_layer2 = Linear(128, 1)

    def reset_parameters(self):
        self.linear_layer1.reset_parameters()
        self.linear_layer2.reset_parameters()
        for i in self.rel_graph_convs:
            i.reset_parameters()

    def forward(self, data):
        num_nodes = len(data.x)
        edge_index_dr, edge_type_dr = dropout_adj(data.edge_index, data.edge_type,\
                                p=0.2, num_nodes=num_nodes, training=self.training)

        out = data.x
        h = []
        for conv in self.rel_graph_convs:
            out = conv(out, edge_index_dr, edge_type_dr)
            out = torch.tanh(out)
            h.append(out)
        h = torch.cat(h, 1)
        h = [h[data.x[:, 0] == True], h[data.x[:, 1] == True]]
        g = torch.cat(h, 1)
        out = self.linear_layer1(g)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear_layer2(out)
        out = out[:,0]
        return out

model = IGMC()

#Trainning
LR = 1e-3
EPOCHS = 80
BATCH_SIZE = 50
LR_DECAY_STEP = 20
LR_DECAY_VALUE = 10

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2)

model.to(device)
model.reset_parameters()
opt = Adam(model.parameters(), lr=LR, weight_decay=0)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss_all = 0
    for train_batch in train_loader:
        opt.zero_grad()
        train_batch = train_batch.to(device)
        y_pred = model(train_batch)
        y_true = train_batch.y
        train_loss = F.mse_loss(y_pred, y_true)
        train_loss.backward()
        train_loss_all += train_loss.item() * train_batch.num_graphs
        opt.step()
    train_loss_all /= len(train_loader.dataset)

    if epoch % LR_DECAY_STEP == 0:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] / LR_DECAY_VALUE
    print('epoch', epoch,'; train loss', train_loss_all)

model.to(device)
model.eval()
test_loss = 0
for test_batch in test_loader:
    test_batch = test_batch.to(device)
    with torch.no_grad():
        y_pred = model(test_batch)
    y_true = test_batch.y
    test_loss += F.mse_loss(y_pred, y_true, reduction='sum')
mse_loss = test_loss.item() / len(test_loader.dataset)

print('test loss', mse_loss)