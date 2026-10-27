class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        vocab = [word for text in texts for word in text.lower().split()]
        unique_vocab = set(vocab)
        sorted_vocab = sorted(list(unique_vocab))
        self.word_to_id = {self.pad_token:0,self.unk_token:1,self.bos_token:2,self.eos_token:3}

        for index,word in enumerate(sorted_vocab):
            self.word_to_id[word]= index+4
        self.vocab_size = len(self.word_to_id)

        for key,value in self.word_to_id.items():
            self.id_to_word[value]=key
        
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        token_ids = []
        words = text.lower().split()

        for word in words:
            word_id = self.word_to_id.get(word, 1)
            token_ids.append(word_id)

        return token_ids
        
            
            
                   

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        words = [self.id_to_word.get(i,"<UNK>") for i in ids]
        text = " ".join(words)
        return text 