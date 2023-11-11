import gensim
import pandas
import numpy


def parse_string_list(s):
    s = s.strip()
    s = s.strip('[]')
    s = s.strip()

    result = []
    current_string = ''
    in_string = False
    for c in s:
        if in_string:
            if c == '\'':
                result.append(current_string)
                current_string = ''
                in_string = False
            else:
                current_string += c
        else:
            if c == '\'':
                in_string = True
    return result


data_file = input('Data path: ')
result_file = input('Result path: ')

print('Loading data...')
data = pandas.read_csv(data_file)
data = data.drop('Unnamed: 0', axis='columns')

print('Loading gensim model...')
model = gensim.models.KeyedVectors.load_word2vec_format('model65.bin', binary=True)

processed = 0


def get_mean_vector(lst):
    global processed
    processed += 1
    if processed % 10000 == 0:
        print(f'Processing item #{processed}...')

    vector_size = 100
    if not isinstance(lst, list):
        return numpy.zeros(vector_size)
    return model.get_mean_vector(lst)


print('Processing data...')
text_columns = ['employer_name', 'experience_name', 'key_skills_name',
                'specializations_profarea_name', 'professional_roles_name',
                'lemmaized_wo_stopwords_raw_description', 'lemmaized_wo_stopwords_raw_branded_description',
                'name_clean', 'employer_industries']

for text_column in text_columns:
    data[text_column] = data[text_column].str.split().apply(get_mean_vector)

print('Saving data...')
data.to_csv(result_file)
