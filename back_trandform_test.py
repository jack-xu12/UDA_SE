import torch
from transformers import MarianMTModel, MarianTokenizer

torch.cuda.empty_cache()

en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").cuda()

fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").cuda()

src_text = [
    "Hi how are you?",
    "I lived in Tokyo for 7 months. Knowing the reality of long train commutes, bike rides from the train station, soup stands, and other typical scenes depicted so well, certainly added to my own appreciation for this film which I really, really liked. There are aspects of Japanese life in this film painted with vivid colors but you don't have to speak Japanese to enjoy this movie. Director Suo's tricks were subtle for the most part; I found his highlighting the character called Tamako Tamura with a soft filter, making her sublime, a tiny bit contrived but most of the directors tricks were so gentle that I was fully pulled in and just danced with his characters. Or cried. Or laughed aloud. Wonderful. A+."
    "I lived in Tokyo for 7 months. Knowing all the long train trips, the bicycle rides from the station, the tables and other typical scenes, so well illustrated, added to my own appreciation of this film what I really enjoyed. In this film there are aspects of the Japanese life painted with bright colors but there is no need to speak Japanese to appreciate this movie. Suo's tricks were tricky for the majority; I found that he put forth the character named Tamako Tamura with a soft filter, which made him sublime, a little bit contorted, but most of the directors' tricks were so soft that I was completely shot in and just danced with his characters. Or wept. Or laughed aloud. Wonderful. A+."
]

translated_tokens = en_fr_model.generate(
    **{k: v.cuda() for k, v in en_fr_tokenizer(src_text, return_tensors="pt", padding=True, max_length=512).items()},
    do_sample=True,
    top_k=10,
    temperature=2.0,
)
in_fr = [en_fr_tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

bt_tokens = fr_en_model.generate(
    **{k: v.cuda() for k, v in fr_en_tokenizer(in_fr, return_tensors="pt", padding=True, max_length=512).items()},
    do_sample=True,
    top_k=10,
    temperature=2.0,
)
in_en = [fr_en_tokenizer.decode(t, skip_special_tokens=True) for t in bt_tokens]