import numpy as np
import pandas as pd
# import csv
import datetime
# import itertools
import networkx as nx
import math
import sys

class Game():
    def __init__(self, L, r, K, adj_list):
        self.L = L
        self.r = r
        self.K = K
        self.N = L*L
        # self.num_detail = 2 * (1+self.num_neighbors)
        
        # state of x is 0 or 1; cooperator: 0, defector: 1
        self.state = np.random.randint(0, 2, self.N)
        # self.neighbors = np.zeros((self.N, self.num_neighbors), dtype=int)


        # for n in range(self.N):
        #     self.neighbors[n,:]= np.array(adj_list[n])
        
        # payoff of x is determined by pair of state of x and y,  CC CD DC DD 
        self._payoff_matrix = np.array([[1, -r], [1+r, 0]])
        
        self.adj_list = adj_list.copy()

        # # mean payoff of x is determined by payoff list and detailed state of x
        # self._mean_list = np.zeros(self.num_detail)
        # for state in range(2):
        #     for defectors in range(1 + self.num_neighbors):
        #         detail = (1 + self.num_neighbors) * state + defectors
        #         mean = (self.num_neighbors - defectors) * self._payoff_list[state][0] + defectors * self._payoff_list[state][1]
        #         self._mean_list[detail] = mean

        # # probability of x copying y's action
        # self._prob_change_list = np.zeros((self.num_detail, self.num_detail))
        # for detail_x in range(self.num_detail):
        #     for detail_y in range(self.num_detail):
        #         px = self._mean_list[detail_x]
        #         py = self._mean_list[detail_y]
        #         self._prob_change_list[detail_x, detail_y] = self._prob_fermi(px-py)

        # # probability distribution of x choosing a neighbor
        # self._prob_choose_list = np.zeros((self.num_detail, self.num_detail, self.num_detail, self.num_detail, 4))
        # for d1 in range(self.num_detail):
        #     for d2 in range(self.num_detail):
        #         for d3 in range(self.num_detail):
        #             for d4 in range(self.num_detail):
        #                 self._prob_choose_list[d1,d2,d3,d4,:] = self._softmax(self._mean_list[[d1,d2,d3,d4]])


        # detailed states of player n, considering the number of defectors among neighbors
        # self.details = np.zeros(self.N, dtype=int)
        # self.mean = np.zeros(self.N)
        # for n in range(self.N):
        #     self._update(n)

        self.mean = np.zeros(self.N)
        for i in range(self.N):
            self._update_mean(i)

    def _update_mean(self, n):
        self.mean[n]= np.mean(self._payoff_matrix[self.state[n],:][self.state[self.adj_list[n]]])

    # update detailed state and mean payoff of player n
    # detailed state is 0 - 9 determined by state of n and # of defectors among neighbors 
    def _update_state(self, n, s):
        self.state[n] = s
        self._update_mean(n)
        for i in self.adj_list[n]:
            self._update_mean(i)

    # dp = px - py
    # Note: if K is too small, take care of sign of K
    # We can consider case 0 < K << 1 but not -1 << K <0
    def _prob_fermi(self, dp, eps = 0.0001):
        if self.K < -eps or eps < self.K:
            return 1 / (1 + np.exp(dp / self.K))
        elif dp > 0.:
            return 0.
        else:
            return 1.

    # def _softmax(self, array):
    #     return np.exp(self.w*array) / np.sum(np.exp(self.w*array))

    def single_step(self):
        # first palyer is chosen
        n_first = np.random.randint(0, self.N)
        neighbors = self.adj_list[n_first]
        # first player chooces second player
        # n_second = np.random.choice(neighbors, p=self._softmax(self.mean[neighbors]))
        n_second = np.random.choice(neighbors)
        # change cannot happen
        if self.state[n_first] == self.state[n_second]:
            pass
        # change may happen
        else:
            # first player chooses whether to copy second player's action
            dp = self.mean[n_first] - self.mean[n_second]
            prob_change_strtgegy = self._prob_fermi(dp)
            if np.random.rand() < prob_change_strtgegy:
                s = 1 if self.state[n_first]==0 else 0
                self._update_state(n_first, s)

    def ratio(self):
        return 1 - self.state.sum()/self.N
    
    def ave_payoff(self):
        return self.mean.mean()
    
    def append_df(self, df, iteration, step):
        cols = call_cols()
        dt_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        record = pd.Series([self.L, self.r, self.K, iteration, step, self.ratio(), self.ave_payoff(), dt_now], index=cols)
        df = df.append(record, ignore_index=True)
        return df

    def display(self, iteration, step):
        print("L", self.L, "r:", self.r, "K:", self.K, "iter:", iteration, "step:", (step+1) // (self.N),
            "ratio:", self.ratio(), "ave_payoff:", self.ave_payoff())
    
def call_cols():
    return ["L", "r", "K", "iterations", "steps", "ratio", "ave_payoff", "datetime"]
        

def main():
    args = sys.argv # r, L, K.logspace(,,)[,], degree, seed
    np.random.seed(32)
    r = float(args[1])
    L = int(args[2]) # 32
    # K_list = np.logspace(float(args[3]), float(args[4]), int(args[5])) # np.logspace(-3., 2., 51)
    K_list = [np.linspace(float(args[3]), float(args[4]), int(args[5]))[int(args[6])]]
    # w = 0.
    iterations = 50
    total_steps = 200
    print_steps = 1
    
    degree = int(args[7])
    seed = int(args[8]) # 42
    print("r:", r, "L:", L, "K_list:", K_list, "degree:", degree, "seed:", seed)

    # G = nx.generators.random_graphs.random_regular_graph(d=degree, n=L*L, seed=seed)
    G = nx.grid_2d_graph(L, L, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    adj_list = nx.to_dict_of_lists(G)

    cols = call_cols()
    df = pd.DataFrame(index=[], columns=cols)

    for K in K_list:
        for iteration in range(iterations):
            game = Game(L, r, K, adj_list)
            game.display(iteration, step=0)
            df = game.append_df(df, iteration, step=0)
            for step in range(total_steps * game.N):
                game.single_step()
                if (step+1) % (print_steps*game.N) == 0:
                    game.display(iteration, step)
                    df = game.append_df(df, iteration, (step+1) // (game.N))
                    if game.state.sum() in [0, game.N]:
                        break
    dt_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = str(dt_now) + "_L" + str(L)  + "_r" + str(r) + "_K" + str(K_list[0]) + "_sq" + "_deg" + str(degree) + "_seed" + str(seed) + ".csv"
    df.to_csv(file_name)

if __name__ == "__main__":
    main()