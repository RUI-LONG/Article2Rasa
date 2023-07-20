import re

class Punctuation:
    def __init__(self):
        self.query = ""
        self.emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
        )

    def set_query(self, query: str):
        self.query = query

    def remove_duplicate_punctuation(self):
        self.query = re.sub(r"[\?？,，\.\!]+(?=[\?？,，\.\!])", "", self.query)

    def remove_emoji(self):
        self.query =  self.emoji_pattern.sub(r"", self.query)
   
    def remove_punctuation(self):
        # punctuation = """＂＃,＄％＆$%&丶:＇（）()＊＋－／：；＜＝＞＠@［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
        punctuation = "$%&(){}^"
        self.query = self.query.translate(
            self.query.maketrans("", "", punctuation)
        )

    def replace_punctuation(self):
        self.query = self.query.replace("？", "?")
    
    def preocess(self):
        self.remove_emoji() 
        self.remove_punctuation()
        self.remove_duplicate_punctuation()
        # self.replace_punctuation()

