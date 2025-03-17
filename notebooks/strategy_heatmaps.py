import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import golden
from scipy.spatial import distance



class HotCold2(nn.Module):
    """Layers for a Hot-Cold strategy

    As described in:

    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """
    def __init__(self, in_channels=2, num_hidden1=15):
        super(HotCold2, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)


class HotCold3(nn.Module):
    """Two layers for a Hot-Cold strategy

    Related to the model described in:

    Muyesser, N.A., Dunovan, K. & Verstynen, T., 2018. Learning model-based
    strategies in simple environments with hierarchical q-networks. , pp.1–29. A
    vailable at: http://arxiv.org/abs/1801.06689.
    """
    def __init__(self, in_channels=2, num_hidden1=100, num_hidden2=25):
        super(HotCold3, self).__init__()
        self.fc1 = nn.Linear(in_channels, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return self.fc3(x)
    

def init_strategist(num_hidden1, num_hidden2):
    """Create a Wythoff's game strategist"""

    num_hidden1 = int(num_hidden1)
    num_hidden2 = int(num_hidden2)
    if num_hidden2 > 0:
        model = HotCold3(2, num_hidden1=num_hidden1, num_hidden2=num_hidden2)
    else:
        model = HotCold2(2, num_hidden1=num_hidden1)

    return model


def load_stumbler(model, opponent, load_model):
    state = th.load(load_model)
    model = state["stumbler_player_dict"]
    opponent = state["stumbler_opponent_dict"]

    return model, opponent


def load_strategist(model, load_model):
    """Override model with parameters from file"""
    state = th.load(load_model)
    model.load_state_dict(state["strategist_model_dict"])

    return model


def expected_value(m, n, model, default_value=0.0):
    """Estimate the max value of each board position"""

    values = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            board = tuple(flatten_board(create_board(i, j, m, n)))
            try:
                v = model[board].max()
                values[i, j] = v
            except KeyError:
                values[i, j] = default_value

    return values


def create_board(i, j, m, n):
    """Create a binary board, with position (i, j) marked."""
    board = np.zeros((m, n))
    board[i, j] = 1.0

    return board


def flatten_board(board):
    m, n = board.shape
    return board.reshape(m * n)


def create_bias_board(m, n, strategist_model, default=0.0):
    """"Sample all positions' value in a strategist model"""
    bias_board = th.ones((m, n), dtype=th.float) * default

    with th.no_grad():
        for i in range(m):
            for j in range(n):
                coords = th.tensor([i, j], dtype=th.float)
                bias_board[i, j] = strategist_model(coords)

    return bias_board


def plot_wythoff_board(board,
                       vmin= -1, #-1.5,
                       vmax=  1, #1.5,
                       plot=False,
                       path=None,
                       height=2,
                       width=3,
                       cbar=False,
                       name='wythoff_board.png'):
    """Plot the board"""

    fig, ax = plt.subplots(figsize=(width, height))  # Sample figsize in inches
    im = ax.imshow(board, cmap='bwr', vmin=vmin, vmax=vmax)
    # ax = sns.heatmap(board,
    #                  linewidths=3,
    #                  center=0,
    #                  vmin=vmin,
    #                  vmax=vmax,
    #                  ax=ax)
    if cbar: fig.colorbar(im) #, orientation='horizontal', pad=0.1)

    # Save an image?
    if path is not None:
        #plt.savefig(os.path.join(path, name))
        plt.savefig(path + name)

    if plot:
        plt.show()
        plt.pause(0.01)

    plt.close('all')

ctr_path = '../data/wythoff/exp13/'
img_path = '../data/wythoff/exp14/'
rep_path = '../data/wythoff/exp14_replay/'

ctr_vals_sum = np.zeros((15,15))
img_vals_sum = np.zeros((50,50))
rep_vals_sum = np.zeros((50,50))

for run in range(100):
    ctr_file = ctr_path + f'run_{run+1}.pytorch'
    img_file = img_path + f'run_{run+1}.pytorch'
    rep_file = rep_path + f'run_{run+1}.pytorch'
    
    player = None
    opponent = None
    player, opponent = load_stumbler(player, opponent, ctr_file)
    ctr_vals = expected_value(15, 15, player)
    ctr_vals_sum += ctr_vals
    
    state = th.load(img_file)
    num_hidden1 = state["num_hidden1"]
    num_hidden2 = state["num_hidden2"]
    strategist = init_strategist(num_hidden1, num_hidden2)
    strategist = load_strategist(strategist, img_file)
    img_vals = create_bias_board(50, 50, strategist).numpy()
    img_vals_sum += img_vals

    state = th.load(img_file)
    num_hidden1 = state["num_hidden1"]
    num_hidden2 = state["num_hidden2"]
    recaller = init_strategist(num_hidden1, num_hidden2)
    recaller = load_strategist(recaller, rep_file)
    rep_vals = create_bias_board(50, 50, recaller).numpy()
    rep_vals_sum += rep_vals

plot_wythoff_board(ctr_vals_sum / 100, plot=True, height=5, width=1.5)#,
                       # path=tensorboard,
                       # name='player_max_values.png')
plot_wythoff_board(0 - (img_vals_sum / 100), plot=True, height=5, width=5)
plot_wythoff_board(0 - (rep_vals_sum / 100), plot=True, height=5, width=5)
plot_wythoff_board(0 - ((img_vals_sum - rep_vals_sum) / 100), plot=True, height=3, width=3, cbar=True)



def create_cold_board(m, n, default=0.0, cold_value=1):
    """Create a (m, n) binary board with cold moves as '1'"""
    cold_board = np.ones((m, n)) * default
    for k in range(m - 1):
        mk = int(k * golden)
        nk = int(k * golden**2)
        if (nk < m) and (mk < n):
            cold_board[mk, nk] = cold_value
            cold_board[nk, mk] = cold_value

    return cold_board

cold_board_50x50 = create_cold_board(50, 50, default=-1, cold_value=1)

# print(cold_board_50x50[:5,:5])
# print(np.array(img_vals_sum/100)[:5,:5])
# print(np.array(rep_vals_sum/100)[:5,:5])
print()
print(distance.cosine(np.array(cold_board_50x50)[:15,:15].flatten(), np.array(img_vals_sum / 100)[:15,:15].flatten()))
print(distance.cosine(np.array(cold_board_50x50)[:15,:15].flatten(), np.array(rep_vals_sum / 100)[:15,:15].flatten()))
print()

cold_board_50x50 = create_cold_board(50, 50, default=0, cold_value=1)
# print(np.array(img_vals_sum/100)[:5,:5][cold_board_50x50[:5,:5].astype(bool)])


def avg_val_at_op_moves_by_board_size(vals, cold_board, size):
    return np.mean(np.array(vals)[:size,:size][cold_board[:size,:size].astype(bool)])

avg_val_by_board_size_img = []
avg_val_by_board_size_rep = []

for size in range(1, 51):
    avg_val_by_board_size_img.append(avg_val_at_op_moves_by_board_size(img_vals_sum / 100, cold_board_50x50, size))
    avg_val_by_board_size_rep.append(avg_val_at_op_moves_by_board_size(rep_vals_sum / 100, cold_board_50x50, size))

print(avg_val_by_board_size_img)
print(avg_val_by_board_size_rep)

fig, ax = plt.subplots()
plt.plot(avg_val_by_board_size_img, label='imagination')
plt.plot(avg_val_by_board_size_rep, label='replay')
plt.title('Average strategist value at optimal moves by board size')
plt.xlabel('Board size')
plt.ylabel('Mean strategist value at optimal moves')
plt.legend(title='agent')
plt.show()




