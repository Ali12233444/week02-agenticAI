# In types ka use neeche list, dictionary aur function return type samjhane ke liye hai.
from typing import Dict, List, Tuple

# Hugging Face tokenizers library ke ready-made models, pre-tokenizers aur trainers.
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


TRAINING_CORPUS = [
    "Artificial intelligence helps people solve problems.",
    "Machine learning models learn patterns from data.",
    "Deep learning uses neural networks.",
    "Natural language processing works with human language.",
    "Tokenization splits text into smaller tokens.",
    "Tokens can be characters words or subwords.",
    "A tokenizer assigns an integer ID to each token.",
    "I am learning Python programming.",
    "Programmers write programs using programming languages.",
    "Playing played player players play games.",
    "Learning learned learner learners learn new skills.",
    "Happy happiness unhappy unhappiness.",
    "Low lower lowest new newer newest.",
    "Hello world welcome to artificial intelligence.",
    "Pakistan Karachi language Urdu English.",
]

# Har tokenizer ka result: (tokens ki list, IDs ki list, vocabulary size).
TokenizationData = Tuple[List[str], List[int], int]


def _simple_tokenize(tokens: List[str]) -> TokenizationData:
    # Token ko uska pehla milne wala integer ID dene ke liye dictionary.
    vocabulary: Dict[str, int] = {}

    for token in tokens:
        if token not in vocabulary:
            # Naya token hamesha current vocabulary length ko next ID banata hai: 0, 1, 2...
            vocabulary[token] = len(vocabulary)

    # Har token ko uski dictionary wali ID me convert karta hai; repeat token ki ID repeat hoti hai.
    token_ids = [vocabulary[token] for token in tokens]
    # Unique tokens ki tadaad hi is simple tokenizer ki vocabulary size hai.
    return tokens, token_ids, len(vocabulary)


def character_level_tokenize(text: str) -> TokenizationData:
    """Split text into individual characters."""
    # Python list(text) spaces aur punctuation samait har character ko alag token banata hai.
    return _simple_tokenize(list(text))


def word_level_tokenize(text: str) -> TokenizationData:
    """Split text into whitespace-separated words."""
    # split() sirf whitespace par todta hai; punctuation word ke sath reh sakti hai.
    return _simple_tokenize(text.split())


def _encode_subword_tokenizer(
    text: str,
    tokenizer: Tokenizer,
    trainer,
) -> TokenizationData:
    # Model training corpus aur current text dono se apni token vocabulary seekhta hai.
    tokenizer.train_from_iterator(
        TRAINING_CORPUS + [text],
        trainer=trainer,
    )
    # Trained model current text ko subword tokens aur unke IDs me encode karta hai.
    encoded = tokenizer.encode(text, add_special_tokens=False)
    # Model ki complete learned vocabulary size return hoti hai, sirf current text ke unique tokens nahi.
    return encoded.tokens, encoded.ids, tokenizer.get_vocab_size()


def bpe_tokenize(text: str) -> TokenizationData:
    """Train and use a standard BPE tokenizer."""
    # BPE frequent character/subword pairs ko merge karke tokens banata hai; unknown token [UNK] hai.
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # Input ko whitespace boundaries par pre-tokenize karta hai, phir BPE in words ko further tod sakta hai.
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # Maximum 180 entries, aur pair/token ko kam az kam 2 dafa milna zaroori hai.
    trainer = trainers.BpeTrainer(
        vocab_size=180,
        min_frequency=2,
        special_tokens=["[UNK]"],
        show_progress=False,
    )
    return _encode_subword_tokenizer(text, tokenizer, trainer)


def wordpiece_tokenize(text: str) -> TokenizationData:
    """Train and use a WordPiece tokenizer."""
    # WordPiece bhi subwords banata hai; word ke baad aane wale pieces ## prefix lete hain.
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    # Pehle words alag karta hai; WordPiece phir har word ko subwords me todta hai.
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # Maximum 180 vocabulary entries aur low-frequency items ko ignore karta hai.
    trainer = trainers.WordPieceTrainer(
        vocab_size=180,
        min_frequency=2,
        special_tokens=["[UNK]"],
        continuing_subword_prefix="##",
        show_progress=False,
    )
    return _encode_subword_tokenizer(text, tokenizer, trainer)


def unigram_tokenize(text: str) -> TokenizationData:
    """Train and use a Unigram tokenizer."""
    # Unigram possible subword pieces me se probability-based best segmentation choose karta hai.
    tokenizer = Tokenizer(models.Unigram())
    # Metaspace spaces ko special marker ke zariye preserve karta hai.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    # Unigram vocabulary ka target 180 hai; final size training data/rules ki wajah se kam ho sakta hai.
    trainer = trainers.UnigramTrainer(
        vocab_size=180,
        special_tokens=["[UNK]"],
        unk_token="[UNK]",
        show_progress=False,
    )
    return _encode_subword_tokenizer(text, tokenizer, trainer)


def byte_level_bpe_tokenize(text: str) -> TokenizationData:
    """Train and use a byte-level BPE tokenizer."""
    # BPE model bytes/byte characters se start karta hai, is liye unusual characters bhi handle ho jate hain.
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # Text ko byte-level units me todta hai; yahan spaces ko pehle word boundary ki tarah nahi hataya jata.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    # 350 target vocabulary hai; byte alphabet pehle se add hone ki wajah se size doosron se different hai.
    trainer = trainers.BpeTrainer(
        vocab_size=350,
        min_frequency=2,
        special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    return _encode_subword_tokenizer(text, tokenizer, trainer)