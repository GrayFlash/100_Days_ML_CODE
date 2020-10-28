import numpy as np

class L_layered_NN:


    def initialize_parameters(nn_architecture, seed=3):
        np.random.seed(seed)
        
        # parameters stores weights and biases in form of dictionary
        parameters = {}
        number_of_layers = len(nn_architecture)
        for l in range(1, number_of_layers):
            parameters['W'+ str(l)] = np.random.randn(
                nn_architecture[l]["layer_size"],
                nn_architecture[l-1]["layer_size"]
            )*0.01
            
            parameters["b"+str(l)] = np.zeros((nn_architecture[l]["layer_size"],1))
            
        return parameters

    # Activation Functions

    def sigmoid(Z):
        return 1/(1 + np.exp(-Z))

    def relu(Z):
        return np.maximum(0, Z)

    def sigmoid_backward(dA, Z):
        S = sigmoid(Z)
        dS = S*(1-S)
        return dA*dS

    def relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z<=0] = 0
        return dZ

    ## Forward function for L-Model

    def L_model_forward(X, parameters, nn_architecture):
        
        forward_cache = {}
        number_of_layers = len(nn_architecture)
        A = X
        
        for l in range(1, number_of_layers):
            A_prev = A
            W = parameters["W"+str(l)]
            b = parameters["b"+str(l)]
            
            activation = nn_architecture[l]["activation"]
            
            Z, A = linear_activation_forward(A_prev, W, b, activation)
            forward_cache['Z'+str(l)] = Z
            forward_cache['A'+str(l)] = A
            
        AL = A
        return AL, forward_cache

        def linear_activation_forward(A_prev, W, b, activation):
            Z = linear_forward(A_prev, W, b)
            
            if activation == "sigmoid":
                A = sigmoid(Z)
            elif activation == "relu":
                A = relu(Z)
                
            return Z, A

    def linear_forward(A, W, b):
        return np.dot(W, A) + b

    def compute_cost(AL, Y):
        m = Y.shape[1]
        
        loprobs = np.multiply(np.log(AL), Y) + np.multiply(1-np.log(AL), 1-Y)
        
        cost = np.sum(logprobs)/m
        
        cost = np.squeeze(cost)
        
        return cost

    def L_model_backward(AL, Y, parameters, forward_cache, nn_architecture):
        grads = {}
        number_of_layers = len(nn_architecture)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 
        
        #Initialisation of backpropagation
        
        dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        dA_prev = dAL
        
        for l in reversed(range(1, number_of_layers)):
            
            dA_curr = dA_prev
            
            activation = nn_architecture[l]["activation"]
            W_curr = parameters['W' + str(l)]
            Z_curr = forward_cache['Z' + str(l)]
            A_prev = forward_cache['A' + str(l-1)]
            
            dA_prev, dW_curr, db_curr = linear_activation_backward(dA_curr, Z_curr, 
                                                                A_prev, W_curr, activation)
            
            grads["dW"+str(l)] = dW_curr
            grads["db"+str(l)] = db_curr
        return grads

    def linear_activation_backwards(dA, Z, A_prev, W, activation):
        
        if activation=="relu":
            dZ = relu_backward(dA, Z)
            dA_prev, dW, db = linear_backward(dZ, A_prev, W)
            
        elif activation=="sigmoid":
            dZ = sigmoid_backward(dA, Z)
            dA_prev, dW, db = linear_backward(dZ, A_prev, W)
            
        return dA_prev, dW, db

    def linear_backward(dZ, A_prev, W):
        
        m = A_prev.shape[1]
        
        dW = np.dot(dZ, A_prev.T)
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def update_parameters(parameters, grads, learning_rate):
        
        L = len(parameters)
        
        for l in range(1, L):
            parameters['W'+str(l)] -= learning_rate*grads["dW"+str(l)]
            parameters['b'+str(l)] -= learning_rate*grads["db"+str(l)]
            
        return parameters

    def L_layer_model(X, Y, nn_architecture, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        np.random.seed(1)
        # keep track of cost
        costs = []
        
        # Parameters initialization.
        parameters = initialize_parameters(nn_architecture)
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, forward_cache = L_model_forward(X, parameters, nn_architecture)
            
            # Compute cost.
            cost = compute_cost(AL, Y)
        
            # Backward propagation.
            grads = L_model_backward(AL, Y, parameters, forward_cache, nn_architecture)
    
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
                    
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))

            costs.append(cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        return parameters