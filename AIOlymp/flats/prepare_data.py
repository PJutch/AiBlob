import pandas


def prepare_data(data: pandas.DataFrame) -> pandas.DataFrame:
    data['agreement_date'] = pandas.to_datetime(data['agreement_date'])
    data['rooms_4'] = data['rooms_4'].replace('>=4', 4).replace('студия', 0)
    data['interior_cat'] = data['interior_cat'].astype('int')
    return data.drop('location_flash_mean_mean', axis=1)
