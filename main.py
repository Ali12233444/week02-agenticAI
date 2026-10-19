from dataclasses import dataclass
from typing import List

# Tokenization functions ko functions.py se import karke yahan application me use kiya ja raha hai.
from functions import (
    byte_level_bpe_tokenize,
    bpe_tokenize,
    character_level_tokenize,
    unigram_tokenize,
    word_level_tokenize,
    wordpiece_tokenize,
)


@dataclass #constructor automatically bnane k liye
class TokenizationResult:
    # Ek method ka complete output ek structured object me rakha jata hai.
    category: str
    tokens: List[str]
    token_ids: List[int]
    vocabulary_size: int

    @property #kahi function ko call krna ho tw srf name pass hota hai brackets sath nhi lagti
    def token_count(self) -> int:
        # Current text me total tokens; repeated tokens bhi count hote hain.
        return len(self.tokens)

    @property
    def unique_token_count(self) -> int:
        # Token IDs ka set duplicates hata deta hai, is liye unique token IDs count hoti hain.
        return len(set(self.token_ids))


class TokenizationApp:
    # User ke numeric choices ko readable method names se map karta hai.
    METHODS = {
        "1": "Character-level",
        "2": "Word-level",
        "3": "Subword: BPE",
        "4": "Subword: WordPiece",
        "5": "Subword: Unigram (SentencePiece-style)",
        "6": "Byte-level BPE",
    }

    def tokenize(self, text: str, choice: str) -> TokenizationResult:
        # Invalid menu choice ko tokenizer call se pehle reject karta hai.
        if choice not in self.METHODS:
            raise ValueError("Please select a valid method.")

        # Khali ya sirf spaces wala input process nahi hota.
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        # Selected number ka display name result me save hota hai.
        category = self.METHODS[choice]

        # Har choice ko uski imported function se connect karta hai.
        tokenizers = {
            "1": character_level_tokenize,
            "2": word_level_tokenize,
            "3": bpe_tokenize,
            "4": wordpiece_tokenize,
            "5": unigram_tokenize,
            "6": byte_level_bpe_tokenize,
        }
        # Selected function call hoti hai; yahin tokens, IDs aur vocabulary size milti hai.
        tokens, token_ids, vocabulary_size = tokenizers[choice](text)

        # Raw output ko TokenizationResult object me pack karke caller ko return karta hai.
        return TokenizationResult(
            category=category,
            tokens=tokens,
            token_ids=token_ids,
            vocabulary_size=vocabulary_size,
        )

    @staticmethod
    def display_result(result: TokenizationResult) -> None:
        # Result ke tokens aur unki matching IDs ko row-by-row print karta hai.
        print("\n" + "=" * 64)
        print(f"METHOD: {result.category}")
        print("=" * 64)
        print(f"{'No.':<6} {'Token':<42} {'Token ID':<10}")
        print("-" * 64)

        for number, (token, token_id) in enumerate(
            zip(result.tokens, result.token_ids),
            start=1,
        ):
            # repr() makes spaces, tabs and newlines visible.
            print(f"{number:<6} {repr(token):<42} {token_id:<10}")

        print("-" * 64)
        print(f"Tokens             : {result.tokens!r}")
        print(f"Token IDs          : {result.token_ids}")
        print(f"Total token count  : {result.token_count}")
        print(f"Unique token count : {result.unique_token_count}")
        print(f"Vocabulary size    : {result.vocabulary_size}")
        print("=" * 64)

    def run(self) -> None:
        # Main interactive loop: menu dikhata hai, input leta hai aur selected method chalata hai.
        print("\nTOKENIZATION ANALYZER")

        while True:
            print("\nAvailable methods:")
            for number, name in self.METHODS.items():
                print(f"{number}. {name}")

            print("7. Compare all methods")
            print("0. Exit")

            choice = input("\nSelect an option: ").strip()

            if choice == "0":
                print("Program closed.")
                break

            if choice not in self.METHODS and choice != "7":
                print("Invalid option. Please choose 0–7.")
                continue

            text = input("Enter your text: ")

            if not text.strip():
                print("Please enter non-empty text.")
                continue

            selected_methods = (
                # Choice 7 par sab methods; warna sirf user ka selected method.
                list(self.METHODS) if choice == "7" else [choice]
            )

            for method in selected_methods:
                try:
                    # Har selected method ke liye tokenize -> result -> display flow.
                    result = self.tokenize(text, method)
                    self.display_result(result)
                except Exception as error:
                    print(
                        f"\nError in {self.METHODS[method]}: {error}"
                    )


def main() -> None:
    app = TokenizationApp()

    try:
        app.run()
    except (KeyboardInterrupt, EOFError):
        print("\nProgram closed.")


if __name__ == "__main__":
    main()