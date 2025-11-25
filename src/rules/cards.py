"""
Card domain: representations for cards, hands, and basic operations.

This module defines the core data structures used throughout the modeling system.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set
from enum import Enum
import random


class Suit(Enum):
    """The four suits in a standard deck."""
    CLUBS = "CLUBS"
    DIAMONDS = "DIAMONDS"
    HEARTS = "HEARTS"
    SPADES = "SPADES"


class Rank(Enum):
    """The 13 ranks in a standard deck."""
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


# Rank value mapping (for numerical comparisons)
RANK_VALUES = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5,
    Rank.SIX: 6, Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
    Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14
}

# String to Rank mapping
STR_TO_RANK = {r.value: r for r in Rank}

# String to Suit mapping
STR_TO_SUIT = {s.value: s for s in Suit}


@dataclass(frozen=True)
class Card:
    """
    An immutable playing card with a suit and rank.

    Examples:
        >>> Card(Suit.SPADES, Rank.ACE)
        Card(SPADES, A)
        >>> Card(Suit.HEARTS, Rank.KING)
        Card(HEARTS, K)
    """
    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        """Unicode representation: rank + suit symbol."""
        suit_symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.HEARTS: "♥",
            Suit.SPADES: "♠"
        }
        return f"{self.rank.value}{suit_symbols[self.suit]}"

    def __repr__(self) -> str:
        return f"Card({self.suit.name}, {self.rank.value})"

    @staticmethod
    def from_string(s: str) -> 'Card':
        """
        Parse a card from string like "A♠" or "AS" or "ACE_OF_SPADES".

        Args:
            s: String representation of card

        Returns:
            Card object

        Examples:
            >>> Card.from_string("A♠")
            Card(SPADES, A)
            >>> Card.from_string("10H")
            Card(HEARTS, 10)
        """
        # Handle unicode symbols
        suit_map = {"♣": Suit.CLUBS, "♦": Suit.DIAMONDS, "♥": Suit.HEARTS, "♠": Suit.SPADES}
        # Handle letter codes
        letter_map = {"C": Suit.CLUBS, "D": Suit.DIAMONDS, "H": Suit.HEARTS, "S": Suit.SPADES}

        # Try unicode first
        for symbol, suit in suit_map.items():
            if symbol in s:
                rank_str = s.replace(symbol, "").strip()
                return Card(suit, STR_TO_RANK[rank_str])

        # Try letter code (last character)
        if len(s) >= 2 and s[-1] in letter_map:
            suit = letter_map[s[-1]]
            rank_str = s[:-1].strip()
            return Card(suit, STR_TO_RANK[rank_str])

        raise ValueError(f"Cannot parse card from string: {s}")


# Type alias for hands
Hand = List[Card]


def sample_hand(size: int = 6, with_replacement: bool = True) -> Hand:
    """
    Sample a random hand of cards.

    Args:
        size: Number of cards to draw
        with_replacement: If True, allow duplicate cards (default)

    Returns:
        List of cards

    Examples:
        >>> hand = sample_hand(6)
        >>> len(hand)
        6
    """
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]

    if with_replacement:
        return random.choices(deck, k=size)
    else:
        if size > len(deck):
            raise ValueError(f"Cannot sample {size} cards without replacement from deck of {len(deck)}")
        return random.sample(deck, k=size)


def hand_to_string(hand: Hand) -> str:
    """
    Convert hand to readable string.

    Args:
        hand: List of cards

    Returns:
        String representation

    Examples:
        >>> h = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)]
        >>> hand_to_string(h)
        '[A♠, K♥]'
    """
    return "[" + ", ".join(str(c) for c in hand) + "]"


def hand_from_strings(cards: List[str]) -> Hand:
    """
    Parse hand from list of card strings.

    Args:
        cards: List of card strings

    Returns:
        Hand object

    Examples:
        >>> hand_from_strings(["AS", "KH", "QD"])
        [Card(SPADES, A), Card(HEARTS, K), Card(DIAMONDS, Q)]
    """
    return [Card.from_string(c) for c in cards]


# Helper constructors (for convenience in tests/examples)
def H(rank_str: str) -> Card:
    """Hearts card: H("A") → Ace of Hearts"""
    return Card(Suit.HEARTS, STR_TO_RANK[rank_str])

def D(rank_str: str) -> Card:
    """Diamonds card: D("K") → King of Diamonds"""
    return Card(Suit.DIAMONDS, STR_TO_RANK[rank_str])

def S(rank_str: str) -> Card:
    """Spades card: S("Q") → Queen of Spades"""
    return Card(Suit.SPADES, STR_TO_RANK[rank_str])

def C(rank_str: str) -> Card:
    """Clubs card: C("J") → Jack of Clubs"""
    return Card(Suit.CLUBS, STR_TO_RANK[rank_str])


# Color types (derived from suits)
class Color(Enum):
    RED = "RED"
    BLACK = "BLACK"


def suit_to_color(suit: Suit) -> Color:
    """Map suit to color."""
    if suit in {Suit.HEARTS, Suit.DIAMONDS}:
        return Color.RED
    return Color.BLACK


def card_color(card: Card) -> Color:
    """Get color of a card."""
    return suit_to_color(card.suit)


# Alternative color categorizations (for rules r27-r31)
class AltColor1(Enum):
    """Pointy (♠♦) vs Round (♥♣)"""
    POINTY = "POINTY"  # Spades, Diamonds
    ROUND = "ROUND"     # Hearts, Clubs


class AltColor2(Enum):
    """SH (♠♥) vs DC (♦♣)"""
    SH = "SH"  # Spades, Hearts
    DC = "DC"  # Diamonds, Clubs


def suit_to_altcolor1(suit: Suit) -> AltColor1:
    """Map suit to alternative color scheme 1."""
    if suit in {Suit.SPADES, Suit.DIAMONDS}:
        return AltColor1.POINTY
    return AltColor1.ROUND


def suit_to_altcolor2(suit: Suit) -> AltColor2:
    """Map suit to alternative color scheme 2."""
    if suit in {Suit.SPADES, Suit.HEARTS}:
        return AltColor2.SH
    return AltColor2.DC


# Parity (odd/even ranks)
class Parity(Enum):
    ODD = "ODD"
    EVEN = "EVEN"


def rank_parity(rank: Rank) -> Parity:
    """Get parity of rank value."""
    val = RANK_VALUES[rank]
    return Parity.ODD if val % 2 == 1 else Parity.EVEN


if __name__ == "__main__":
    # Quick tests
    print("=== Card Domain Tests ===")

    # Create some cards
    ace_spades = Card(Suit.SPADES, Rank.ACE)
    king_hearts = Card(Suit.HEARTS, Rank.KING)

    print(f"Ace of Spades: {ace_spades}")
    print(f"King of Hearts: {king_hearts}")

    # Sample a hand
    hand = sample_hand(6)
    print(f"\nRandom hand: {hand_to_string(hand)}")

    # Test parsing
    parsed = Card.from_string("A♠")
    print(f"\nParsed 'A♠': {parsed}")

    # Test helpers
    h = [H("A"), D("K"), S("Q"), C("J")]
    print(f"\nConstructed hand: {hand_to_string(h)}")

    # Test colors
    print(f"\nColors: {card_color(H('A'))}, {card_color(S('K'))}")
    print(f"AltColor1: {suit_to_altcolor1(Suit.SPADES)}, {suit_to_altcolor1(Suit.HEARTS)}")
    print(f"Parity: {rank_parity(Rank.ACE)}, {rank_parity(Rank.KING)}")
