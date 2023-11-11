import spacy
from spacy import displacy
from pathlib import Path


def quote(string):
    return '"' + string + '"'


nlp = spacy.load("ru_core_news_sm")

doc = nlp(input())

for token in doc:
    print(f'text={quote(token.text):10} lemma={quote(token.lemma_):10} '
          f'pos={token.pos_:5} tag={token.tag_:5} dep={token.dep_:5} shape={token.shape_:4} '
          f'is_alpha={str(token.is_alpha):5} is_stop={str(token.is_stop):5}')

svg_dep = displacy.render(doc, style="dep")
output_path = Path("dependencies.svg")
output_path.open("w", encoding="utf-8").write(svg_dep)

for ent in doc.ents:
    print(f'text={quote(ent.text):10}, label={ent.label_:3}')

html_ent = displacy.render(doc, style="ent")
with open('entities.html', 'w') as f:
    f.write(html_ent)
