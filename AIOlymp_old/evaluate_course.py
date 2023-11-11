import pandas

df = pandas.read_csv('train_courses.csv').sample(frac=1.0)
train = df[:-500]
cross_test = df[-500:]

course_freq = train['course_id'].value_counts()
base_course_score = list(dict(course_freq).items())

user_courses = {}
for i in train.index:
    user_courses.setdefault(train['user_id'][i], [])
    user_courses[train['user_id'][i]].append(train['course_id'][i])

other_from_course_freq = {}
for user in user_courses:
    for course in user_courses[user]:
        other_from_course_freq.setdefault(course, {})
        for other in user_courses[user]:
            if other == course:
                continue

            other_from_course_freq[course].setdefault(other, 0)
            other_from_course_freq[course][other] += 1

cross_test_user_courses = {}
for i in cross_test.index:
    cross_test_user_courses.setdefault(cross_test['user_id'][i], [])
    cross_test_user_courses[cross_test['user_id'][i]].append(cross_test['course_id'][i])

mean_precision = 0
for user in cross_test_user_courses:
    course_score = base_course_score.copy()

    for i in range(len(course_score)):
        for other in cross_test_user_courses[user]:
            if other in other_from_course_freq:
                if course_score[i][0] in other_from_course_freq[other]:
                    course_score[i] = course_score[i][0],  (course_score[i][1]
                                      * other_from_course_freq[other][course_score[i][0]] / course_freq[other])
    course_score.sort(key=lambda pair: pair[1], reverse=True)

    precision = 0
    K = 3
    for course, score in course_score[:K]:
        if course in cross_test_user_courses[user]:
            precision += 1

    precision /= min(K, len(cross_test_user_courses[user]))
    mean_precision += precision
mean_precision /= len(cross_test_user_courses)
print(mean_precision)
