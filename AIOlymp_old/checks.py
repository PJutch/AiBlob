import pandas


def median(lst):
    return sorted(lst)[len(lst) // 2]


data = pandas.read_csv('train.csv', encoding='UTF-8')
neighbours = []
for row in data.iterrows():
    while row[1]['check_id'] >= len(neighbours):
        neighbours.append([])

    name = row[1]['name']
    if pandas.isna(name):
        continue

    new_name = ''
    prev_space = True
    for char in name.lower().strip():
        if char.isalpha():
            new_name += char
            prev_space = False
        elif char.isspace():
            if not prev_space:
                new_name += char
                prev_space = True
        else:
            if not prev_space:
                new_name += ' '
                prev_space = True
    neighbours[row[1]['check_id']].append(new_name.strip())

neighbours_freq = {}
for neighbour_packet in neighbours:
    for neighbour in neighbour_packet:
        neighbours_freq.setdefault(neighbour, {})
        for neighbour2 in neighbour_packet:
            if neighbour == neighbour2:
                continue

            neighbours_freq[neighbour].setdefault(neighbour2, 0)
            neighbours_freq[neighbour][neighbour2] += 1

for neighbour in neighbours_freq:
    total = sum(neighbours_freq[neighbour].values())
    for neighbour2 in neighbours_freq[neighbour]:
        neighbours_freq[neighbour][neighbour2] /= total

with open('answer2.csv', encoding='UTF-8', mode='w') as f:
    f.write('product, related_product\n')
    for neighbour, frequencies in neighbours_freq.items():
        if frequencies:
            m = median(frequencies.values())
            freq_list = [record[0] for record in frequencies.items() if record[1] >= m]
            for item in freq_list:
                f.write(f'"{neighbour}","{item}"\n')
