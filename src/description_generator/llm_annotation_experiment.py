"""
LLM-based Hand Annotation Experiment

This module implements a blind annotation approach where:
1. Hands are sampled randomly for various rules
2. An LLM describes each hand WITHOUT knowing the rule
3. Descriptions are optimized for discrimination and brevity

The key insight is that descriptions come from observing the hand itself,
not from foreknowledge of the rule - more cognitively plausible.
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
import hashlib

# Import card infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank, sample_hand


@dataclass
class AnnotatedHand:
    """A hand with its blind LLM annotation."""
    hand_id: str  # Unique identifier for this hand
    cards: List[str]  # Card strings like "7♠", "Q♥"
    description: Optional[str] = None  # LLM-generated description
    tokens: Optional[List[str]] = None  # Tokenized description

    def __post_init__(self):
        if self.hand_id is None:
            # Generate ID from card content
            card_str = "_".join(self.cards)
            self.hand_id = hashlib.md5(card_str.encode()).hexdigest()[:8]


@dataclass
class RuleDataset:
    """Dataset of hands for a single rule."""
    rule_id: str
    rule_name: str
    rule_family: str
    positive_hands: List[AnnotatedHand]
    negative_hands: List[AnnotatedHand]

    def all_hands(self) -> List[Tuple[AnnotatedHand, bool]]:
        """Return all hands with their labels."""
        return [(h, True) for h in self.positive_hands] + [(h, False) for h in self.negative_hands]


# ============================================================================
# SELECTED RULES - 20 diverse rules covering all families
# ============================================================================

SELECTED_RULES = [
    # LOCAL family
    ("r1x", "Sorted_by_rank", "Ranks are in non-decreasing order"),
    ("r44x", "Ends_same_suit", "First and last share the suit"),
    ("r45x", "Ends_same_color", "First and last share the color"),

    # COUNT family
    ("r2x", "Has_pair_ranks", "At least one pair (same rank)"),
    ("r3x", "Uniform_color", "All cards have the same color"),
    ("p7x", "Exactly_two_suits", "Exactly two suits appear"),
    ("r43x", "Exactly_one_club", "Exactly one club"),

    # AP family
    ("r5x", "AP_len3_anywhere_anyk", "3-term rank pattern with equal steps"),

    # SCORE family
    ("r42x", "Half_sum_diff_geN", "Left half beats right by at least N points"),

    # HIER family
    ("r12x", "Halves_uniform_color_equal", "Both halves uniform in color (or both not)"),
    ("r26x", "Halves_hearts_presence_equal", "Both halves have heart or neither"),

    # LANG family
    ("r15x", "Well_formed_brackets_by_suit", "Suits form matched brackets"),

    # PAL family
    ("r18x", "Suits_palindrome", "Suits read same forward/back"),
    ("r19x", "Colors_palindrome", "Colors read same forward/back"),

    # COPY family
    ("r21x", "Halves_copy_suits", "Halves have same suit sequence"),

    # ADJ family
    ("r38x", "Adj_same_rank_or_suit", "Neighbors share rank or suit"),
    ("r40x", "Adj_rank_gap_le3", "Neighbors differ by ≤3 ranks"),

    # PARITY family
    ("r47x", "Only_one_odd_rank", "Exactly one odd rank"),
    ("r48x", "Uniform_rank_parity", "All ranks same parity"),

    # CENTER family
    ("r49x", "Halves_radial_nonincreasing", "Outward from center, ranks don't go up"),
]


def sample_balanced_hands(
    rule_predicate: Callable[[Hand], bool],
    n_positive: int = 5,
    n_negative: int = 5,
    hand_size: int = 6,
    max_attempts: int = 10000,
    seed: Optional[int] = None
) -> Tuple[List[Hand], List[Hand]]:
    """
    Sample balanced positive and negative examples for a rule.

    Returns:
        (positive_hands, negative_hands)
    """
    if seed is not None:
        random.seed(seed)

    positives = []
    negatives = []
    attempts = 0

    while (len(positives) < n_positive or len(negatives) < n_negative) and attempts < max_attempts:
        hand = sample_hand(hand_size)
        attempts += 1

        if rule_predicate(hand):
            if len(positives) < n_positive:
                # Check for duplicates
                hand_tuple = tuple((c.rank, c.suit) for c in hand)
                if not any(tuple((c.rank, c.suit) for c in h) == hand_tuple for h in positives):
                    positives.append(hand)
        else:
            if len(negatives) < n_negative:
                hand_tuple = tuple((c.rank, c.suit) for c in hand)
                if not any(tuple((c.rank, c.suit) for c in h) == hand_tuple for h in negatives):
                    negatives.append(hand)

    if len(positives) < n_positive:
        print(f"  Warning: Only found {len(positives)}/{n_positive} positive examples in {max_attempts} attempts")
    if len(negatives) < n_negative:
        print(f"  Warning: Only found {len(negatives)}/{n_negative} negative examples in {max_attempts} attempts")

    return positives, negatives


def hand_to_annotated(hand: Hand, hand_id: Optional[str] = None) -> AnnotatedHand:
    """Convert a Hand to an AnnotatedHand ready for LLM processing."""
    cards = []
    for card in hand:
        suit_char = {'CLUBS': '♣', 'DIAMONDS': '♦', 'HEARTS': '♥', 'SPADES': '♠'}[card.suit.name]
        rank_char = {
            'TWO': '2', 'THREE': '3', 'FOUR': '4', 'FIVE': '5', 'SIX': '6',
            'SEVEN': '7', 'EIGHT': '8', 'NINE': '9', 'TEN': '10',
            'JACK': 'J', 'QUEEN': 'Q', 'KING': 'K', 'ACE': 'A'
        }[card.rank.name]
        cards.append(f"{rank_char}{suit_char}")

    return AnnotatedHand(hand_id=hand_id, cards=cards)


def create_experiment_dataset(
    n_positive: int = 5,
    n_negative: int = 5,
    hand_size: int = 6,
    seed: int = 42
) -> List[RuleDataset]:
    """
    Create the full experiment dataset with hands for all 20 rules.

    Hands are sampled independently of the annotation process.
    """
    from rules.catalogue import get_rule_by_token

    datasets = []

    print("Sampling hands for 20 rules...")
    for i, (token, rule_id, name) in enumerate(SELECTED_RULES):
        print(f"  [{i+1}/20] {rule_id}...")

        try:
            rule = get_rule_by_token(token)
        except ValueError:
            print(f"    Warning: Rule {token} not found in catalogue, skipping")
            continue

        # Sample with rule-specific seed for reproducibility
        rule_seed = seed + hash(rule_id) % 1000
        positives, negatives = sample_balanced_hands(
            rule.predicate,
            n_positive=n_positive,
            n_negative=n_negative,
            hand_size=hand_size,
            seed=rule_seed
        )

        # Convert to annotated hands
        pos_hands = [
            hand_to_annotated(h, f"{rule_id}_pos_{j}")
            for j, h in enumerate(positives)
        ]
        neg_hands = [
            hand_to_annotated(h, f"{rule_id}_neg_{j}")
            for j, h in enumerate(negatives)
        ]

        datasets.append(RuleDataset(
            rule_id=rule_id,
            rule_name=name,
            rule_family=rule.family,
            positive_hands=pos_hands,
            negative_hands=neg_hands
        ))

    return datasets


def save_dataset(datasets: List[RuleDataset], output_path: Path):
    """Save dataset to JSON for annotation."""
    data = {
        "metadata": {
            "n_rules": len(datasets),
            "n_hands_per_rule": len(datasets[0].positive_hands) + len(datasets[0].negative_hands) if datasets else 0,
            "annotation_status": "pending"
        },
        "rules": []
    }

    for ds in datasets:
        rule_data = {
            "rule_id": ds.rule_id,
            "rule_name": ds.rule_name,
            "rule_family": ds.rule_family,
            "positive_hands": [asdict(h) for h in ds.positive_hands],
            "negative_hands": [asdict(h) for h in ds.negative_hands]
        }
        data["rules"].append(rule_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved dataset to {output_path}")


def load_dataset(input_path: Path) -> List[RuleDataset]:
    """Load dataset from JSON."""
    with open(input_path) as f:
        data = json.load(f)

    datasets = []
    for rule_data in data["rules"]:
        datasets.append(RuleDataset(
            rule_id=rule_data["rule_id"],
            rule_name=rule_data["rule_name"],
            rule_family=rule_data["rule_family"],
            positive_hands=[AnnotatedHand(**h) for h in rule_data["positive_hands"]],
            negative_hands=[AnnotatedHand(**h) for h in rule_data["negative_hands"]]
        ))

    return datasets


# ============================================================================
# LLM ANNOTATION PROMPTS
# ============================================================================

ANNOTATION_SYSTEM_PROMPT = """You are a concise card hand descriptor. Your job is to describe card hands in the most discriminating yet brief way possible.

IMPORTANT CONSTRAINTS:
1. Use MINIMAL words - aim for 5-15 words per description
2. Focus on DISTINCTIVE features that separate this hand from random hands
3. Use a CONSISTENT vocabulary - prefer common terms like:
   - "sorted", "pairs", "flush", "run", "alternating", "palindrome"
   - "red", "black", "hearts", "spades", "diamonds", "clubs"
   - "low", "high", "even", "odd", "gap"
4. Do NOT mention the number of cards (always 6)
5. NEVER use "contains" or "includes" - just state features directly

Good examples:
- "all hearts, sorted ascending"
- "pair of 7s, alternating colors"
- "red-black-red-black-red-black"
- "same rank first and last"

Bad examples (too verbose):
- "This hand contains six cards, all of which are hearts and are arranged in ascending order"
- "The cards in this hand include a pair of sevens and the colors alternate between red and black"
"""

ANNOTATION_USER_TEMPLATE = """Describe this 6-card hand concisely:

{cards}

Description (5-15 words):"""


def format_hand_for_llm(hand: AnnotatedHand) -> str:
    """Format a hand for LLM input."""
    return " ".join(hand.cards)


def create_annotation_batch(hands: List[AnnotatedHand], batch_size: int = 10) -> List[str]:
    """Create batched annotation prompts for efficiency."""
    batches = []

    for i in range(0, len(hands), batch_size):
        batch = hands[i:i+batch_size]
        prompt = "Describe each of these 6-card hands concisely (5-15 words each):\n\n"
        for j, hand in enumerate(batch):
            prompt += f"{j+1}. {' '.join(hand.cards)}\n"
        prompt += "\nDescriptions (one per line, numbered):\n"
        batches.append(prompt)

    return batches


# ============================================================================
# ANNOTATION WITH DIFFERENT BACKENDS
# ============================================================================

def annotate_with_claude_api(
    datasets: List[RuleDataset],
    api_key: Optional[str] = None,
    model: str = "claude-3-5-haiku-20241022"
) -> List[RuleDataset]:
    """
    Annotate hands using Claude API.

    Uses Haiku for speed and cost efficiency.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Please install anthropic: pip install anthropic")

    import os
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set ANTHROPIC_API_KEY environment variable.")

    client = Anthropic(api_key=api_key)

    # Collect all hands for batch processing
    all_hands = []
    hand_to_location = {}  # Map hand_id to (dataset_idx, list_name, idx)

    for ds_idx, ds in enumerate(datasets):
        for h_idx, hand in enumerate(ds.positive_hands):
            all_hands.append(hand)
            hand_to_location[hand.hand_id] = (ds_idx, "positive", h_idx)
        for h_idx, hand in enumerate(ds.negative_hands):
            all_hands.append(hand)
            hand_to_location[hand.hand_id] = (ds_idx, "negative", h_idx)

    print(f"Annotating {len(all_hands)} hands with Claude {model}...")

    # Process in batches
    batch_size = 20
    for batch_start in range(0, len(all_hands), batch_size):
        batch_end = min(batch_start + batch_size, len(all_hands))
        batch = all_hands[batch_start:batch_end]

        # Create batch prompt
        prompt = "Describe each of these 6-card hands concisely (5-15 words each). Focus on distinctive features.\n\n"
        for i, hand in enumerate(batch):
            prompt += f"{i+1}. {' '.join(hand.cards)}\n"
        prompt += "\nProvide one description per line, numbered 1-{n}:".format(n=len(batch))

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=ANNOTATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            text = response.content[0].text
            lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

            for i, hand in enumerate(batch):
                if i < len(lines):
                    # Remove numbering if present
                    desc = lines[i]
                    if desc.startswith(f"{i+1}.") or desc.startswith(f"{i+1})"):
                        desc = desc.split('.', 1)[-1].strip() if '.' in desc else desc.split(')', 1)[-1].strip()
                    hand.description = desc
                    hand.tokens = desc.lower().split()

        except Exception as e:
            print(f"  Error in batch {batch_start//batch_size + 1}: {e}")
            continue

        print(f"  Processed {batch_end}/{len(all_hands)} hands")

    # Update datasets
    for hand in all_hands:
        ds_idx, list_name, h_idx = hand_to_location[hand.hand_id]
        if list_name == "positive":
            datasets[ds_idx].positive_hands[h_idx] = hand
        else:
            datasets[ds_idx].negative_hands[h_idx] = hand

    return datasets


def annotate_with_ollama(
    datasets: List[RuleDataset],
    model: str = "qwen2.5-coder:7b"
) -> List[RuleDataset]:
    """
    Annotate hands using local Ollama model.

    Uses qwen2.5-coder by default for good instruction following.
    """
    import subprocess
    import json as json_mod

    # Collect all hands
    all_hands = []
    hand_to_location = {}

    for ds_idx, ds in enumerate(datasets):
        for h_idx, hand in enumerate(ds.positive_hands):
            all_hands.append(hand)
            hand_to_location[hand.hand_id] = (ds_idx, "positive", h_idx)
        for h_idx, hand in enumerate(ds.negative_hands):
            all_hands.append(hand)
            hand_to_location[hand.hand_id] = (ds_idx, "negative", h_idx)

    print(f"Annotating {len(all_hands)} hands with Ollama {model}...")

    # Process one at a time for reliability (batch processing can be flaky with local models)
    for i, hand in enumerate(all_hands):
        cards_str = " ".join(hand.cards)
        prompt = f"""Describe this card hand in EXACTLY 2-3 words. Be extremely terse.

Examples:
- "all hearts"
- "pair, sorted"
- "alternating colors"
- "three clubs"
- "high cards"

Hand: {cards_str}

Description (2-3 words ONLY):"""

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                desc = result.stdout.strip()
                # Clean up - take only first line, remove quotes
                desc = desc.split('\n')[0].strip().strip('"\'')
                # Truncate to max 4 words (allow slight overflow)
                words = desc.split()
                if len(words) > 4:
                    desc = " ".join(words[:3])

                hand.description = desc
                hand.tokens = desc.lower().split()
            else:
                print(f"  Warning: Failed on hand {i+1}: {result.stderr[:100]}")
                hand.description = "annotation failed"
                hand.tokens = []

        except subprocess.TimeoutExpired:
            print(f"  Warning: Timeout on hand {i+1}")
            hand.description = "timeout"
            hand.tokens = []
        except Exception as e:
            print(f"  Warning: Error on hand {i+1}: {e}")
            hand.description = "error"
            hand.tokens = []

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(all_hands)} hands")

    # Update datasets
    for hand in all_hands:
        ds_idx, list_name, h_idx = hand_to_location[hand.hand_id]
        if list_name == "positive":
            datasets[ds_idx].positive_hands[h_idx] = hand
        else:
            datasets[ds_idx].negative_hands[h_idx] = hand

    print(f"  Processed {len(all_hands)}/{len(all_hands)} hands")
    return datasets


def annotate_with_local_rules(datasets: List[RuleDataset]) -> List[RuleDataset]:
    """
    Simple rule-based annotation as a baseline.

    This is NOT as good as LLM annotation but can be used for testing.
    """

    def describe_hand_simple(cards: List[str]) -> str:
        """Generate a simple description using basic features."""
        suits = [c[-1] for c in cards]
        ranks = [c[:-1] for c in cards]
        colors = ['red' if s in '♦♥' else 'black' for s in suits]

        # Check various features
        features = []

        # Uniform suit
        if len(set(suits)) == 1:
            suit_names = {'♣': 'clubs', '♦': 'diamonds', '♥': 'hearts', '♠': 'spades'}
            features.append(f"all {suit_names[suits[0]]}")

        # Uniform color
        elif len(set(colors)) == 1:
            features.append(f"all {colors[0]}")

        # Count suits
        else:
            n_suits = len(set(suits))
            features.append(f"{n_suits} suits")

        # Check for pairs
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        pairs = [r for r, c in rank_counts.items() if c >= 2]
        if pairs:
            features.append(f"pair of {pairs[0]}s")

        # Check sorted
        rank_order = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                      '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}
        values = [rank_order.get(r, 0) for r in ranks]
        if values == sorted(values):
            features.append("sorted ascending")
        elif values == sorted(values, reverse=True):
            features.append("sorted descending")

        # Check alternating colors
        alternating = all(colors[i] != colors[i+1] for i in range(len(colors)-1))
        if alternating:
            features.append("alternating colors")

        # First and last
        if suits[0] == suits[-1]:
            features.append("same suit ends")
        if colors[0] == colors[-1]:
            features.append("same color ends")

        return ", ".join(features[:3]) if features else "mixed"

    print("Annotating hands with rule-based descriptions...")

    for ds in datasets:
        for hand in ds.positive_hands:
            hand.description = describe_hand_simple(hand.cards)
            hand.tokens = hand.description.lower().split()
        for hand in ds.negative_hands:
            hand.description = describe_hand_simple(hand.cards)
            hand.tokens = hand.description.lower().split()

    return datasets


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_vocabulary(datasets: List[RuleDataset]) -> Dict:
    """Analyze the vocabulary used in annotations."""
    all_tokens = []
    all_descriptions = []

    for ds in datasets:
        for hand in ds.positive_hands + ds.negative_hands:
            if hand.tokens:
                all_tokens.extend(hand.tokens)
            if hand.description:
                all_descriptions.append(hand.description)

    # Token frequency
    token_freq = {}
    for t in all_tokens:
        token_freq[t] = token_freq.get(t, 0) + 1

    # Description length stats
    desc_lengths = [len(d.split()) for d in all_descriptions]

    return {
        "total_hands": len(all_descriptions),
        "unique_tokens": len(token_freq),
        "total_tokens": len(all_tokens),
        "avg_tokens_per_desc": len(all_tokens) / len(all_descriptions) if all_descriptions else 0,
        "avg_desc_length": sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0,
        "top_tokens": sorted(token_freq.items(), key=lambda x: -x[1])[:30],
        "token_distribution": token_freq
    }


def compute_discrimination(datasets: List[RuleDataset]) -> Dict:
    """
    Compute how well descriptions discriminate between positive and negative examples.

    For each rule, check if positive hands share tokens that negative hands don't have.
    """
    results = {}

    for ds in datasets:
        pos_tokens = set()
        neg_tokens = set()

        for hand in ds.positive_hands:
            if hand.tokens:
                pos_tokens.update(hand.tokens)

        for hand in ds.negative_hands:
            if hand.tokens:
                neg_tokens.update(hand.tokens)

        # Tokens unique to positives
        unique_pos = pos_tokens - neg_tokens
        # Tokens unique to negatives
        unique_neg = neg_tokens - pos_tokens
        # Shared tokens
        shared = pos_tokens & neg_tokens

        results[ds.rule_id] = {
            "unique_positive_tokens": list(unique_pos),
            "unique_negative_tokens": list(unique_neg),
            "shared_tokens": list(shared),
            "discrimination_ratio": len(unique_pos) / (len(shared) + 1)  # Avoid div by zero
        }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the full experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based hand annotation experiment")
    parser.add_argument("--sample-only", action="store_true", help="Only sample hands, don't annotate")
    parser.add_argument("--use-claude", action="store_true", help="Use Claude API for annotation")
    parser.add_argument("--use-ollama", action="store_true", help="Use local Ollama for annotation")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5-coder:7b", help="Ollama model to use")
    parser.add_argument("--output", type=str, default="llm_annotation_dataset.json")
    parser.add_argument("--n-pos", type=int, default=5, help="Number of positive examples per rule")
    parser.add_argument("--n-neg", type=int, default=5, help="Number of negative examples per rule")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / "results_llm_annotation"
    output_path = output_dir / args.output

    # Step 1: Create dataset
    print("=" * 60)
    print("LLM Annotation Experiment")
    print("=" * 60)

    datasets = create_experiment_dataset(
        n_positive=args.n_pos,
        n_negative=args.n_neg,
        seed=args.seed
    )

    print(f"\nGenerated {len(datasets)} rule datasets")
    total_hands = sum(len(ds.positive_hands) + len(ds.negative_hands) for ds in datasets)
    print(f"Total hands: {total_hands}")

    if args.sample_only:
        save_dataset(datasets, output_path)
        print("\nSampling complete. Dataset saved (no annotation).")
        return

    # Step 2: Annotate
    if args.use_claude:
        datasets = annotate_with_claude_api(datasets)
    elif args.use_ollama:
        datasets = annotate_with_ollama(datasets, model=args.ollama_model)
    else:
        datasets = annotate_with_local_rules(datasets)

    # Step 3: Analyze
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    vocab_analysis = analyze_vocabulary(datasets)
    print(f"\nVocabulary Statistics:")
    print(f"  Unique tokens: {vocab_analysis['unique_tokens']}")
    print(f"  Avg tokens per description: {vocab_analysis['avg_tokens_per_desc']:.1f}")
    print(f"  Top 15 tokens: {[t[0] for t in vocab_analysis['top_tokens'][:15]]}")

    discrimination = compute_discrimination(datasets)
    avg_disc = sum(d['discrimination_ratio'] for d in discrimination.values()) / len(discrimination)
    print(f"\nDiscrimination (avg): {avg_disc:.2f}")

    # Step 4: Save
    save_dataset(datasets, output_path)

    # Save analysis
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            "vocabulary": {k: v for k, v in vocab_analysis.items() if k != "token_distribution"},
            "discrimination": discrimination
        }, f, indent=2)
    print(f"Saved analysis to {analysis_path}")

    # Print sample descriptions
    print("\n" + "=" * 60)
    print("Sample Descriptions")
    print("=" * 60)

    for ds in datasets[:3]:
        print(f"\n{ds.rule_id} ({ds.rule_family}):")
        print(f"  Rule: {ds.rule_name}")
        if ds.positive_hands and ds.positive_hands[0].description:
            print(f"  + {ds.positive_hands[0].cards} → \"{ds.positive_hands[0].description}\"")
        if ds.negative_hands and ds.negative_hands[0].description:
            print(f"  - {ds.negative_hands[0].cards} → \"{ds.negative_hands[0].description}\"")


if __name__ == "__main__":
    main()
