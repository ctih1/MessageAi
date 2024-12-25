class Splitter:
    def __init__(self):
        self.sentences = []
    def split(self,story:list) -> list:
        for sentences in story:
            self.sentences.extend(sentences.split("."))

