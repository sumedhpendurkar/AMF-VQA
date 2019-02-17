import spacy

class TextPreProcess():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, sentence):
        token_list = []
        doc = self.nlp(sentence)
        for token in doc:
            token_list.append(token.text)
        return token_list




if __name__ == "__main__":
    a = TextPreProcess()
    print(a.tokenize("This is a template sentence"))