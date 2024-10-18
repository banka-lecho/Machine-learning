class SVMClassifier:
  def __init__(self, kernel='linear', C=0, degree=3, gamma=0.1, learning_rate=0.001, epochs=10):
    self.kernel = kernel
    self.C = C 
    self.degree = degree
    self.gamma = gamma 
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.w = None
    self.b = 0
    self.errors = []
    self.alpha = None
    self.train_er = []
    self.test_er = []

  def rbf_kernel(self, X1, X2):
        X1_square = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_square = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        return np.exp(-self.gamma * (X1_square + X2_square - 2 * np.dot(X1, X2.T)))

  
  def polynomial_kernel(self, X1, X2):
        return (1 + np.dot(X1, X2.T)) ** self.degree


  def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)


  def eval_kernel(self, X1, X2):
        if self.kernel == 'linear':
          return self.linear_kernel(X1, X2)

        elif self.kernel == 'rbf':
          return self.rbf_kernel(X1, X2)
          
        elif self.kernel == 'polynomial':
          return self.polynomial_kernel(X1, X2)

  def fit(self, X, y, X_test, y_test):
        self.w = np.zeros(X.shape[1])
        self.alpha = np.zeros(X.shape[0])
        K = self.eval_kernel(X, X)

        for ep in range(self.epochs):
            for i in range(X.shape[0]):
              prediction = np.sum(self.alpha * y * K[i]) - self.b
              if y[i] * prediction < 1:
                  self.alpha[i] += self.learning_rate

              if prediction >= 1:
                  self.w = np.dot(X.T, self.alpha * y)
              else:
                  self.w = np.clip(np.dot(X.T, self.alpha * y) - 2 * self.C * self.w, -0.1, 1e10)
              self.b = np.mean(y - np.dot(X, self.w))

              self.train_er.append(np.mean((self.predict(X) - y) ** 2))
              mean_f1 = f1_score(y_test, self.predict(X_test))
              self.test_er.append(mean_f1)
        self.support_vectors = X[self.alpha > 10e-5]

  def predict(self, X):
    return np.sign(np.dot(X, self.w) + self.b)
