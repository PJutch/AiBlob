import pandas

train = pandas.read_csv('train_courses.csv').sample(frac=1.0)

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

test = pandas.read_csv('sample_submission_course.csv').sample(frac=1.0)
for i in test.index:
    user_courses.setdefault(test['user_id'][i], [])

print('user_id,course_id_1,course_id_2,course_id_3')
for i in test.index:
    user = test['user_id'][i]
    print(user, end='')

    course_score = base_course_score.copy()
    for i in range(len(course_score)):
        for other in user_courses[user]:
            if other in other_from_course_freq:
                if course_score[i][0] in other_from_course_freq[other]:
                    course_score[i] = course_score[i][0],  (course_score[i][1]
                                      * other_from_course_freq[other][course_score[i][0]] / course_freq[other])
    course_score.sort(key=lambda pair: pair[1], reverse=True)

    output = 0
    for course, _ in course_score:
        if course not in user_courses[user]:
            print(',' + str(course), end='')
            output += 1
            if output == 3:
                break
    print()
