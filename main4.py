import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


class MLP:

    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(5,),
                             activation='relu',
                             solver='adam',
                             learning_rate='adaptive',
                             max_iter=10000,
                             learning_rate_init=0.001,
                             alpha=0.0001)

    def fit(self, X_train, y_train):
        self.model = self.model.fit(X_train, y_train)

    def get_weights(self):
        return self.model.coefs_

    def get_mse(self, X, y):
        y_pred = self.model.predict(X)
        return sum((y - y_pred) ** 2)/len(X)


class PSO:

    def __init__(self, mlp, pop_size, max_iter, stop_fitness, c1, c2, inertia, vmax):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.stop_fitness = stop_fitness
        self.c1 = c1
        self.c2 = c2
        self.inertia = inertia
        self.vmax = vmax
        self.mlp = mlp
        self.w = self.mlp.get_weights()
        n_w = self.w[0].flatten().shape[0] + self.w[1].flatten().shape[0]
        self.layer_margin  = self.w[0].flatten().shape[0]
        self.population = np.random.rand(pop_size, n_w)
        self.velocities = np.random.rand(pop_size, n_w)*(vmax) - np.random.rand(pop_size, n_w)*(vmax)
        self.pbest = np.random.rand(pop_size, n_w)
        self.pbest_f = np.full(pop_size, np.inf)
        self.gbest = None
        self.gbest_f = np.inf

    def update_mlp_weights(self, particle):
        hidden_layer = particle[:self.layer_margin]
        output_layer = particle[self.layer_margin:]
        hidden_layer = hidden_layer.reshape(self.w[0].shape)
        output_layer = output_layer.reshape(self.w[1].shape)
        self.w[0] = hidden_layer
        self.w[1] = output_layer

    def fitness(self, X, y):
        return self.mlp.get_mse(X,y)

    def evaluate_population(self, X, y):
        for i, particle in enumerate(self.population):
            self.update_mlp_weights(particle)
            f = self.fitness(X, y)
            if f < self.pbest_f[i]:
                self.pbest_f[i] = f
                self.pbest[i] = particle.copy()
            if f < self.gbest_f:
                self.gbest_f = f
                self.gbest = particle.copy()

    def update_velocities(self):
        velocity_inertia = self.velocities*self.inertia
        local_influence = self.c1*np.random.rand(*self.population.shape)*(self.pbest - self.population)
        global_influence = self.c2*np.random.rand(*self.population.shape)*(self.gbest - self.population)
        self.velocities = velocity_inertia + local_influence + global_influence
        self.velocities = np.clip(self.velocities, -self.vmax, self.vmax)

    def update_positions(self):

        self.population = self.inertia + self.velocities

    def fit(self, X_train, y_train, X_test, y_test):

        for i in range(self.max_iter):
            self.evaluate_population(X_train, y_train)
            self.update_mlp_weights(self.gbest)
            f_test = self.fitness(X_test, y_test)
            print("f test:", f_test)
            print("iter:", i+1)
            if f_test <= self.stop_fitness:
                break

            self.update_velocities()
            self.update_positions()
        return i+1

def main():
    np.random.seed(2)

    scaler = MinMaxScaler((-1, 1))
    data = np.loadtxt('data.txt')
    data = scaler.fit_transform(data) # Uncomment to scale data [-1, 1]
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0:7], data[:, -1], test_size=0.2)
    iters = []
    mlp_mse_list = []
    pso_mse_list = []
    for i in range(10):
        mlp = MLP()
        mlp.fit(X_train, y_train)
        mlp_mse = mlp.get_mse(X_test, y_test)

        pso = PSO(mlp, 1000, 100, mlp_mse, c1=2.5, c2=0.6, inertia=1.05, vmax=20)
        j = pso.fit(X_train, y_train, X_test, y_test)
        iters.append(j)
        print("Required fitness reached after {j} iterations.".format(j=j))
        print("Gbest:", pso.fitness(X_test, y_test))
        print("Mlp:", mlp_mse)
        mlp_mse_list.append(mlp_mse)
        pso_mse_list.append(pso.gbest_f)






if __name__ == "__main__":
    main()
