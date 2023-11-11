import pandas
import numpy
import scipy
import pickle

if __name__ == '__main__':
    print('Loading data...')
    data = pandas.read_csv('train_compressed.csv')
    data = data.drop('Unnamed: 0', axis=1)
    data.drop(data[(data['status'] == 'P') | (data['status'] == 'B')].index)
    data = data.drop(['join_request_date', 'status'], axis=1)

    # matrix = data.head(len(data) // 1024).pivot(columns='community_id', values='status').replace(numpy.NAN, False)

    print('Building sparse matrix...')
    k1, k2, k3 = numpy.unique(data, return_inverse=True, return_index=True)
    rows, cols = k3.reshape(data.shape).T
    matrix = scipy.sparse.coo_matrix((numpy.ones(rows.shape, float), (rows, cols))).tocsc()

    print('Factorizing...')
    u, s, v = scipy.sparse.linalg.svds(matrix, k=100)

    print('Saving...')
    with open('community_v3.pth', 'wb') as f:
        pickle.dump(u @ numpy.diag(s), f)
    with open('customer_v3.pth', 'wb') as f:
        pickle.dump(v, f)
