from transformers import pipeline

text = '''This was a masterpiece. Not completely faithful to the books, 
          but enthralling from beginning to end. 
          Might be my favorite of the three.'''

pipe = pipeline('sentiment-analysis', 'en_classifier2')
print(pipe(text))
