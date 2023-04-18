    dim=X.shape[0]
    Results=0
    for i in range(dim):
        Results=Results+np.sum(X[0:i+1])**2

    return Results