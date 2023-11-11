import os
import pathlib
import spacy
import numpy

nlp = spacy.load('ru_core_news_lg')


def load_to_transform_dict(file):
    examples = {}
    with open(file, encoding='UTF-8') as f:
        for line in f:
            frm, to, *flags = line.split()
            if 'no_forms' not in flags:
                examples[frm] = to
    return examples


def load_all_dict(file):
    examples = {}
    with open(file, encoding='UTF-8') as f:
        for line in f:
            frm, to, *flags = line.split()
            examples[frm] = to
    return examples


def load_examples(directory):
    examples = {}
    with os.scandir(directory) as suffixes:
        for entry in suffixes:
            if not entry.name.startswith('.') and entry.is_file():
                path = directory / pathlib.Path(entry.name)
                examples[path.stem] = load_all_dict(path)
    return examples


def to_matrix(words):
    vectors = []
    for word in words:
        vectors.append(nlp(word).vector)
    return numpy.asarray(vectors)


def mean_offset(frm, to):
    return (to - frm).mean(axis=0)


def to_transforms(examples):
    transformations = {}
    for affix, affix_examples in examples.items():
        transformations[affix] = mean_offset(to_matrix(affix_examples.keys()),
                                             to_matrix(affix_examples.values()))
    return transformations


def cosine_similarity(v1, v2):
    return numpy.dot(v1, v2) / numpy.linalg.norm(v1) / numpy.linalg.norm(v2)


def transform(dictionary, transformation):
    matrix = to_matrix(dictionary.values())

    transformed_matrix = matrix + transformation.reshape((1, transformation.size)).repeat(matrix.shape[0], axis=0)
    most_similar = nlp.vocab.vectors.most_similar(transformed_matrix, n=10)

    transformed = {}
    for index, key in enumerate(dictionary.keys()):
        for hashed in most_similar[0][index]:
            if cosine_similarity(nlp.vocab.vectors[hashed], matrix[index]) < 0.5:
                break

            word = nlp.vocab.strings[hashed]
            if word != dictionary[key]:
                transformed[key] = word
                break
    return transformed


def transform_all(all_dictionary, to_transform_dictionary, suffixes, prefixes):
    result = all_dictionary.copy()

    for suffix, transformation in suffixes.items():
        transformed = transform(to_transform_dictionary, transformation)
        for key, value in transformed.items():
            result.setdefault(key + suffix, value)

    for prefix, transformation in prefixes.items():
        transformed = transform(to_transform_dictionary, transformation)
        for key, value in transformed.items():
            result.setdefault(prefix + key, value)

    return result


if __name__ == '__main__':
    suffix_transforms = to_transforms(load_examples('suffix_examples'))
    prefix_transforms = to_transforms(load_examples('prefix_examples'))
    basic_dict = load_all_dict('basic_dict.txt')
    to_transform_dict = load_to_transform_dict('basic_dict.txt')
    print(transform_all(basic_dict, to_transform_dict, suffix_transforms, prefix_transforms))
