import pandas
import pickle

if __name__ == '__main__':
    data = pandas.read_csv('train_df.tsv', sep='\t')
    data = data.drop('Unnamed: 0', axis=1)

    community_ids = data['community_id'].unique()
    community_id_map = {v: i for i, v in enumerate(community_ids)}
    data['community_id'] = data['community_id'].astype('category').cat.rename_categories(community_id_map)

    customer_ids = data['customer_id'].unique()
    customer_id_map = {v: i for i, v in enumerate(customer_ids)}
    data['customer_id'] = data['customer_id'].astype('category').cat.rename_categories(customer_id_map)

    customer_columns = ['customer_id', 'status', 'join_request_date']
    community_data = data.drop(customer_columns, axis=1).drop_duplicates().set_index('community_id')

    community_data_columns = ['description', 'customers_count', 'messages_count',
                              'type', 'region_id', 'themeid', 'business_category', 'business_parent']
    join_data = data.drop(community_data_columns, axis=1)

    join_data.to_csv('train_compressed.csv')
    community_data.to_csv('train_communities.csv')

    with open('community_id_map.pth', 'wb') as f:
        pickle.dump(community_id_map, f)

    with open('customer_id_map.pth', 'wb') as f:
        pickle.dump(customer_id_map, f)
