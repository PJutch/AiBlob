import pandas
import numpy
import scipy
import pickle
from timeit import default_timer as timer

if __name__ == '__main__':
    start_time = timer()
    print('Loading data...')
    data = pandas.read_csv('train_compressed.csv')
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop(data[(data['status'] == 'P') | (data['status'] == 'B')].index)
    data = data.drop(['join_request_date', 'status'], axis=1)

    # matrix = data.head(len(data) // 1024).pivot(columns='community_id', values='status').replace(numpy.NAN, False)

    k1, k2, k3 = numpy.unique(data, return_inverse=True, return_index=True)
    rows, cols = k3.reshape(data.shape).T
    is_member = scipy.sparse.coo_matrix((numpy.ones(rows.shape, float), (rows, cols))).tocsc()

    print('Loading artifacts...')
    with open('community_v2.pth', 'rb') as f:
        community_v: numpy.ndarray = pickle.load(f)

    with open('customer_v2.pth', 'rb') as f:
        customer_v = pickle.load(f)

    with open('community_id_map.pth', 'rb') as f:
        community_id_map = pickle.load(f)

    with open('customer_id_map.pth', 'rb') as f:
        customer_id_map = pickle.load(f)

    print('Loading query...')
    customers = pandas.read_csv('test_customer_ids.csv')
    customers['customer_id'] = customers['customer_id'].astype('category').cat.rename_categories(customer_id_map)

    query_v = customer_v.T[customers['customer_id']]

    batch_size = 5000
    res = numpy.zeros((query_v.shape[0], 7))
    for i in range(0, query_v.shape[0], batch_size):
        print(f'Processing #{i}...')

        end = min(i + batch_size, query_v.shape[0])

        customer_id = customers['customer_id'][i:end]
        weights = community_v @ query_v[i:end].T

        mask = is_member.T[customer_id].toarray().astype('bool').T

        weights[mask] = -1000
        res[i:end] = numpy.argpartition(weights, 7, axis=0)[:7].T

    print('Saving...')
    with open('prediction2.csv', 'w') as f:
        f.write('customer_id,community_id_1,community_id_2,community_id_3,community_id_4,community_id_5,'
                'community_id_6,community_id_7\n')
        for row in res:
            for v in row:
                f.write(str(int(v)) + '\n')

    print(timer() - start_time)
