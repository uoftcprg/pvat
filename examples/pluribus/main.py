from collections.abc import Collection, Iterable, Iterator, Sequence
from collections import defaultdict
from functools import partial
from glob import glob
from itertools import combinations
from operator import eq
from os import system
from pathlib import Path
from sys import path as syspath
from typing import Any, TypeAlias

from pokerkit import Card, HandHistory, Statistics
from tqdm import tqdm
import numpy as np

syspath.append(str(Path(__file__).parent.parent.parent))

from pvat import MIVAT, LinearValueFunction  # noqa: E402

PHH_DATASET_PATH: Path = Path(__file__).parent / 'phh-dataset'
PHH_PATHS: tuple[Path, ...] = tuple(
    map(
        Path,
        glob(str(PHH_DATASET_PATH / 'data' / 'pluribus' / '*' / '*.phh')),
    ),
)
BIG_BLIND_UNIT: int = 1000
POST_FLOP_INPUT_FILE_PATH: Path = Path(__file__).parent / 'input.txt'
POKERCMD_PATH: Path = Path(__file__).parent / 'pokercmd'
POKER_HAND_STRENGTHS_PATH: Path = POKERCMD_PATH / 'poker-hand-strengths'
POKER_HAND_STRENGTHS_TIMEOUT: float = 0.002
POST_FLOP_OUTPUT_FILE_PATH: Path = Path(__file__).parent / 'output.txt'
H: TypeAlias = tuple[int, int, frozenset[Card]]
A: TypeAlias = frozenset[Card]
P: TypeAlias = None
CHANCE: None = None
PSA_DATASET_PATH: Path = Path(__file__).parent / 'psa-dataset'
PRE_FLOP_INPUT_FILE_PATH: Path = (
    PSA_DATASET_PATH
    / 'data'
    / 'hand-strengths'
    / 'standard'
    / 'preflop'
    / 'input.txt'
)
PRE_FLOP_OUTPUT_FILE_PATH: Path = (
    PSA_DATASET_PATH
    / 'data'
    / 'hand-strengths'
    / 'standard'
    / 'preflop'
    / 'output.txt'
)
FLOP_INPUT_FILE_PATH: Path = (
    PSA_DATASET_PATH
    / 'data'
    / 'hand-strengths'
    / 'standard'
    / 'flop'
    / 'input.txt'
)
FLOP_OUTPUT_FILE_PATH: Path = (
    PSA_DATASET_PATH
    / 'data'
    / 'hand-strengths'
    / 'standard'
    / 'flop'
    / 'output.txt'
)
ZERO_SUM: bool = True
PLURIBUS_NAME: str = 'Pluribus'
BASIC_ESTIMATE_NAME: str = 'Basic'
MIVAT_ESTIMATE_NAME: str = 'MIVAT w/ Linear Value Function'
AIVAT_ESTIMATE_NAME: str = 'AIVAT (Brown and Sandholm)'
AIVAT_ESTIMATE_MEAN: int = 48
AIVAT_ESTIMATE_STDERR: int = 25


def load_hand_histories() -> tuple[HandHistory, ...]:
    print('Loading hand histories...')

    hand_histories = []

    for path in tqdm(PHH_PATHS):
        with open(path, 'rb') as file:
            hand_history = HandHistory.load(file)

        deck_cards = set(hand_history.create_game().deck)
        board_cards = list[Card]()
        hole_cards = [
            list[Card]() for _ in range(len(hand_history.starting_stacks))
        ]
        statuses = []
        pots = []
        actions = []
        dealable_cards = []

        for state, action in hand_history.iter_state_actions():
            if action is None:
                continue

            words = action.split()

            if '#' in words:
                words = words[:words.index('#')]

            cards = set()

            match words:
                case 'd', 'db', raw_cards:
                    cards = set(Card.parse(raw_cards))

                    assert len(cards) in {3, 1, 1}

                    board_cards.extend(cards)
                case 'd', 'dh', player, raw_cards:
                    cards = set(Card.parse(raw_cards))
                    index = int(player[1:]) - 1

                    assert not hole_cards[index]
                    assert not index or hole_cards[index - 1]
                    assert len(cards) == 2
                    assert action == hand_history.actions[index]

                    hole_cards[index].extend(cards)

            if cards:
                assert words[0] == 'd'

                statuses.append(np.array(state.statuses))
                pots.append(state.total_pot_amount)
                actions.append(cards)
                dealable_cards.append(deck_cards.copy())

                assert cards <= deck_cards

                deck_cards -= cards

        assert len(board_cards) in {0, 3, 4, 5}
        assert all(map(partial(eq, 2), map(len, hole_cards)))
        assert (
            6
            <= len(actions)
            == len(dealable_cards)
            == 6 + max(0, len(board_cards) - 2)
            <= 9
        )
        assert list(map(set, hole_cards)) == actions[:6]

        hand_history.user_defined_fields['_board_cards'] = board_cards
        hand_history.user_defined_fields['_hole_cards'] = hole_cards
        hand_history.user_defined_fields['_statuses'] = statuses
        hand_history.user_defined_fields['_pots'] = pots
        hand_history.user_defined_fields['_actions'] = actions
        hand_history.user_defined_fields['_dealable_cards'] = dealable_cards

        hand_histories.append(hand_history)

    return tuple(hand_histories)


def load_statistics(
        hand_histories: Iterable[HandHistory],
) -> dict[str, Statistics]:
    print('Loading statistics...')

    return Statistics.from_hand_history(*hand_histories)


def get_conversion_rate(hand_histories: Iterable[HandHistory]) -> float:
    hand_history = next(iter(hand_histories))

    assert hand_history.blinds_or_straddles is not None

    return BIG_BLIND_UNIT / hand_history.blinds_or_straddles[1]


def serialize(cards: Collection[Card]) -> str:
    if cards:
        raw_cards = ''.join(sorted(map(repr, cards)))
    else:
        raw_cards = '-'

    return raw_cards


def deserialize(raw_cards: str) -> frozenset[Card]:
    if raw_cards == '-':
        cards = frozenset[Card]()
    else:
        cards = frozenset(Card.parse(raw_cards))

    return cards


def write_input_file(hand_histories: Iterable[HandHistory]) -> bool:
    print('Writing input.txt file...')

    inputs = defaultdict[frozenset[Card], set[frozenset[Card]]](set)

    for hand_history in hand_histories:
        board_cards = hand_history.user_defined_fields['_board_cards']
        hole_cards = hand_history.user_defined_fields['_hole_cards']
        dealable_cards = hand_history.user_defined_fields['_dealable_cards']

        if len(board_cards) >= 4:
            assert len(dealable_cards) >= 8
            assert len(dealable_cards[7]) == 37
            assert (
                set((board_cards[3],))
                == hand_history.user_defined_fields['_actions'][7]
            )
            assert (
                set(board_cards[:3])
                == hand_history.user_defined_fields['_actions'][6]
            )
            assert board_cards[3] in dealable_cards[7]
            assert (
                len(dealable_cards) == 8
                or board_cards[3] not in dealable_cards[8]
            )

            for card in dealable_cards[7]:
                inputs[frozenset(board_cards[:3] + [card])].update(
                    map(frozenset, hole_cards),
                )

        if len(board_cards) == 5:
            assert len(dealable_cards) == 9
            assert len(dealable_cards[8]) == 36
            assert (
                set((board_cards[4],))
                == hand_history.user_defined_fields['_actions'][8]
            )
            assert board_cards[4] in dealable_cards[8]

            for card in dealable_cards[8]:
                inputs[frozenset(board_cards[:4] + [card])].update(
                    map(frozenset, hole_cards),
                )

    items = []

    for key, value in inputs.items():
        items.append((serialize(key), ' '.join(sorted(map(serialize, value)))))

    lines = list(map(' '.join, sorted(items)))
    content = '\n'.join(lines)

    if POST_FLOP_INPUT_FILE_PATH.exists():
        with open(POST_FLOP_INPUT_FILE_PATH) as file:
            status = content != file.read()
    else:
        status = True

    if status:
        with open(POST_FLOP_INPUT_FILE_PATH, 'w') as file:
            file.write(content)

    return status


def make_pokercmd() -> None:
    print('Compiling PokerCMD...')
    system(f'cd {str(POKERCMD_PATH)} && make')


def write_output_file() -> None:
    print('Writing output.txt file...')

    INPUT = f'cat {str(POST_FLOP_INPUT_FILE_PATH)}'
    COMMAND = f'{POKER_HAND_STRENGTHS_PATH} {POKER_HAND_STRENGTHS_TIMEOUT}'
    OUTPUT = str(POST_FLOP_OUTPUT_FILE_PATH)

    system(f'{INPUT} | {COMMAND} > {OUTPUT}')


def get_terminal_value(
        hand_histories: Sequence[HandHistory],
        h: H,
) -> Any:
    i, _, _ = h
    hand_history = hand_histories[i]

    assert h[1] == len(hand_history.user_defined_fields['_actions'])
    assert not h[2]
    assert hand_history.finishing_stacks is not None

    return np.subtract(
        hand_history.finishing_stacks,
        hand_history.starting_stacks,
    )


def get_actions(hand_histories: Sequence[HandHistory], h: H) -> Iterator[A]:
    i, j, _ = h
    hand_history = hand_histories[i]
    action = hand_history.user_defined_fields['_actions'][j]
    dealable_cards = hand_history.user_defined_fields['_dealable_cards'][j]

    assert not h[2]

    for cards in combinations(dealable_cards, len(action)):
        yield frozenset(cards)


def get_probability(hand_histories: Sequence[HandHistory], h: H) -> int:
    assert h[2]

    return 1


def get_made(hand_histories: Sequence[HandHistory], h: H, a: A) -> H:
    i, j, _ = h

    assert not h[2]

    return i, j, a


def get_prefixes(
        hand_histories: Sequence[HandHistory],
        h: H,
) -> Iterator[tuple[H, A]]:
    i, j, _ = h
    hand_history = hand_histories[i]
    actions = hand_history.user_defined_fields['_actions']

    assert j == len(actions)
    assert not h[2]

    for j, action in enumerate(actions):
        yield (i, j, frozenset()), action


def get_player(
        hand_histories: Sequence[HandHistory],
        h: H,
) -> P:
    assert (
        0
        <= h[1]
        < len(hand_histories[h[0]].user_defined_fields['_actions'])
    )
    assert not h[2]

    return CHANCE


def create_mivat(hand_histories: Sequence[HandHistory]) -> MIVAT[H, A, P]:
    return MIVAT(
        partial(get_terminal_value, hand_histories),
        partial(get_actions, hand_histories),
        partial(get_probability, hand_histories),
        partial(get_made, hand_histories),
        partial(get_prefixes, hand_histories),
        partial(get_player, hand_histories),
        CHANCE,
    )


def create_terminal_histories(
        hand_histories: Sequence[HandHistory],
) -> Iterator[H]:
    for i, hand_history in enumerate(hand_histories):
        actions = hand_history.user_defined_fields['_actions']

        yield i, len(actions), frozenset()


def load_hand_strengths(
) -> dict[tuple[frozenset[Card], frozenset[Card]], float]:
    print('Loading hand strengths...')

    inputs = []
    outputs = list[float]()

    for input_file_path, output_file_path in (
            (PRE_FLOP_INPUT_FILE_PATH, PRE_FLOP_OUTPUT_FILE_PATH),
            (FLOP_INPUT_FILE_PATH, FLOP_OUTPUT_FILE_PATH),
            (POST_FLOP_INPUT_FILE_PATH, POST_FLOP_OUTPUT_FILE_PATH),
    ):
        with open(input_file_path) as file:
            for raw_board_cards, *raw_hole_cards in map(str.split, file):
                board_cards = deserialize(raw_board_cards)

                for hole_cards in map(deserialize, raw_hole_cards):
                    inputs.append((board_cards, hole_cards))

        with open(output_file_path) as file:
            outputs.extend(map(float, file.read().split()))

        assert len(inputs) == len(outputs)

    hand_strengths = dict(zip(inputs, outputs))
    hand_strengths[frozenset(), frozenset()] = 0

    return hand_strengths


def extract_features(
        hand_histories: Sequence[HandHistory],
        hand_strengths: dict[tuple[frozenset[Card], frozenset[Card]], float],
        h: H,
) -> Any:
    i, j, k = h
    hand_history = hand_histories[i]
    statuses = hand_history.user_defined_fields['_statuses'][j]
    pot = hand_history.user_defined_fields['_pots'][j]
    actions = hand_history.user_defined_fields['_actions']

    assert j < len(actions)
    assert k

    hole_cards = [set[Card]() for _ in range(6)]
    board_cards = set()

    for a, action in enumerate(actions[:j] + [k]):
        if a < 6:
            hole_cards[a].update(action)
        else:
            board_cards.update(action)

    hand_strength_features = np.empty(6)

    for n in range(6):
        hand_strength_features[n] = hand_strengths.get(
            (frozenset(board_cards), frozenset(hole_cards[n])),
            0,
        )

    hand_strength_features **= statuses.sum()
    features = np.hstack(
        (
            hand_strength_features,
            statuses * hand_strength_features,
            (1 - statuses) * hand_strength_features,
            1,
        ),
    )
    features *= pot
    full_features = np.zeros((9, 19))
    full_features[j] = features

    return full_features.ravel()


def learn_linear_value_function(
        hand_histories: Sequence[HandHistory],
        mivat: MIVAT[H, A, P],
        terminal_histories: Sequence[H],
        hand_strengths: dict[tuple[frozenset[Card], frozenset[Card]], float],
) -> LinearValueFunction[H]:
    print('Learning linear value function...')

    return LinearValueFunction.learn(
        mivat.get_base_term,
        mivat.get_correction_term_sum,
        partial(extract_features, hand_histories, hand_strengths),
        tqdm(terminal_histories),
        ZERO_SUM,
    )


def estimate(
        hand_histories: Sequence[HandHistory],
        mivat: MIVAT[H, A, P],
        terminal_histories: Sequence[H],
        linear_value_function: LinearValueFunction[H],
) -> None:
    print('Making value estimates...')

    for hand_history, terminal_history in tqdm(
            zip(hand_histories, terminal_histories),
            total=len(PHH_PATHS),
    ):
        finishing_stacks = (
            mivat(linear_value_function, terminal_history)
            + hand_history.starting_stacks
        )
        hand_history.finishing_stacks = finishing_stacks.tolist()


def print_value_estimate(name: str, mean: float, stderr: float) -> None:
    print(f'{name}: {mean:+.0f} ± {stderr:.0f} mbb/game')


def main() -> None:
    hand_histories = load_hand_histories()
    statistics = load_statistics(hand_histories)
    unit = get_conversion_rate(hand_histories)
    basic_estimate_mean = statistics[PLURIBUS_NAME].payoff_mean
    basic_estimate_stderr = statistics[PLURIBUS_NAME].payoff_stderr

    if write_input_file(hand_histories):
        make_pokercmd()
        write_output_file()

    mivat = create_mivat(hand_histories)
    terminal_histories = tuple(create_terminal_histories(hand_histories))
    hand_strengths = load_hand_strengths()
    linear_value_function = learn_linear_value_function(
        hand_histories,
        mivat,
        terminal_histories,
        hand_strengths,
    )

    estimate(hand_histories, mivat, terminal_histories, linear_value_function)

    statistics = load_statistics(hand_histories)
    mivat_estimate_mean = statistics[PLURIBUS_NAME].payoff_mean
    mivat_estimate_stderr = statistics[PLURIBUS_NAME].payoff_stderr

    print_value_estimate(
        BASIC_ESTIMATE_NAME,
        basic_estimate_mean * unit,
        basic_estimate_stderr * unit,
    )
    print_value_estimate(
        MIVAT_ESTIMATE_NAME,
        mivat_estimate_mean * unit,
        mivat_estimate_stderr * unit,
    )
    print_value_estimate(
        AIVAT_ESTIMATE_NAME,
        AIVAT_ESTIMATE_MEAN,
        AIVAT_ESTIMATE_STDERR,
    )


if __name__ == '__main__':
    main()
