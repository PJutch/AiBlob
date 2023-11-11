import pandas
import pickle

if __name__ == '__main__':
    with open('community_id_map.pth', 'rb') as f:
        community_id_map = pickle.load(f)
    community_id_map = {v: k for k, v in community_id_map.items()}

    customers = pandas.read_csv('test_customer_ids.csv')

    with open('prediction2.csv', 'r') as f:
        header = f.readline()
        lines = list(map(int, f.readlines()))

    lines = [(customers['customer_id'][i // 7],)
             + tuple(community_id_map[lines[i + j]] for j in range(7)) for i in range(0, len(lines), 7)]
    lines = [','.join(line) + '\n' for line in lines]

    with open('prediction_formatted2.csv', 'w') as f:
        f.write(header)
        f.writelines(lines)
