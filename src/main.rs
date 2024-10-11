use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Rank {
    Two = 2,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

#[derive(Debug, Clone, Copy)]
struct Card {
    rank: Rank,
    suit: Suit,
}

impl Card {
    fn new(rank: Rank, suit: Suit) -> Self {
        Card { rank, suit }
    }
}

struct Deck {
    cards: Vec<Card>,
}

impl Deck {
    fn new() -> Self {
        let mut cards = Vec::with_capacity(52);
        for &suit in &[Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            for rank_value in 2..=14 {
                let rank = match rank_value {
                    2 => Rank::Two,
                    3 => Rank::Three,
                    4 => Rank::Four,
                    5 => Rank::Five,
                    6 => Rank::Six,
                    7 => Rank::Seven,
                    8 => Rank::Eight,
                    9 => Rank::Nine,
                    10 => Rank::Ten,
                    11 => Rank::Jack,
                    12 => Rank::Queen,
                    13 => Rank::King,
                    14 => Rank::Ace,
                    _ => unreachable!(),
                };
                cards.push(Card::new(rank, suit));
            }
        }
        Deck { cards }
    }

    fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.cards.shuffle(&mut rng);
    }

    fn deal(&mut self) -> Option<Card> {
        self.cards.pop()
    }
}

#[derive(Debug)]
struct Player {
    hand: Vec<Card>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum HandRank {
    HighCard(Rank),
    OnePair(Rank),
    TwoPair(Rank, Rank),
    ThreeOfAKind(Rank),
    Straight(Rank),
    Flush(Rank),
    FullHouse(Rank, Rank),
    FourOfAKind(Rank),
    StraightFlush(Rank),
    RoyalFlush,
}

// Function to map HandRank instances to their categories
fn hand_rank_category(hand_rank: &HandRank) -> &'static str {
    match hand_rank {
        HandRank::HighCard(_) => "HighCard",
        HandRank::OnePair(_) => "OnePair",
        HandRank::TwoPair(_, _) => "TwoPair",
        HandRank::ThreeOfAKind(_) => "ThreeOfAKind",
        HandRank::Straight(_) => "Straight",
        HandRank::Flush(_) => "Flush",
        HandRank::FullHouse(_, _) => "FullHouse",
        HandRank::FourOfAKind(_) => "FourOfAKind",
        HandRank::StraightFlush(_) => "StraightFlush",
        HandRank::RoyalFlush => "RoyalFlush",
    }
}

fn is_sequence(mut ranks: Vec<u8>) -> bool {
    ranks.sort_unstable();
    ranks.dedup();

    if ranks.len() != 5 {
        return false;
    }

    let is_regular_straight = ranks[4] - ranks[0] == 4;

    // Special case for wheel straight (A-2-3-4-5)
    let is_wheel_straight = ranks == vec![2, 3, 4, 5, 14];

    is_regular_straight || is_wheel_straight
}

fn get_rank_counts(ranks: &[Rank]) -> HashMap<Rank, u8> {
    let mut counts = HashMap::new();
    for &rank in ranks {
        *counts.entry(rank).or_insert(0) += 1;
    }
    counts
}

fn evaluate_five_card_hand(cards: &[&Card]) -> HandRank {
    let mut ranks: Vec<Rank> = cards.iter().map(|c| c.rank).collect();
    let mut rank_values: Vec<u8> = ranks.iter().map(|&r| r as u8).collect();
    ranks.sort_by(|a, b| b.cmp(a)); // Sort descending
    rank_values.sort_unstable();
    rank_values.dedup();

    let suits: Vec<Suit> = cards.iter().map(|c| c.suit).collect();

    let is_flush = suits.iter().all(|&s| s == suits[0]);

    let is_straight = is_sequence(rank_values.clone());

    if is_flush && is_straight {
        if ranks.contains(&Rank::Ace) && ranks.contains(&Rank::King) {
            return HandRank::RoyalFlush;
        } else {
            return HandRank::StraightFlush(ranks[0]);
        }
    }

    let rank_counts = get_rank_counts(&ranks);

    let counts: Vec<u8> = rank_counts.values().cloned().collect();
    if counts.contains(&4) {
        let rank = *rank_counts
            .iter()
            .find(|&(_, &count)| count == 4)
            .unwrap()
            .0;
        return HandRank::FourOfAKind(rank);
    }

    if counts.contains(&3) && counts.contains(&2) {
        let three_rank = *rank_counts
            .iter()
            .find(|&(_, &count)| count == 3)
            .unwrap()
            .0;
        let two_rank = *rank_counts
            .iter()
            .find(|&(_, &count)| count == 2)
            .unwrap()
            .0;
        return HandRank::FullHouse(three_rank, two_rank);
    }

    if is_flush {
        return HandRank::Flush(ranks[0]);
    }

    if is_straight {
        return HandRank::Straight(ranks[0]);
    }

    if counts.contains(&3) {
        let rank = *rank_counts
            .iter()
            .find(|&(_, &count)| count == 3)
            .unwrap()
            .0;
        return HandRank::ThreeOfAKind(rank);
    }

    let pair_ranks: Vec<Rank> = rank_counts
        .iter()
        .filter(|&(_, &count)| count == 2)
        .map(|(&rank, _)| rank)
        .collect();

    if pair_ranks.len() == 2 {
        return HandRank::TwoPair(pair_ranks[0], pair_ranks[1]);
    } else if pair_ranks.len() == 1 {
        return HandRank::OnePair(pair_ranks[0]);
    }

    HandRank::HighCard(ranks[0])
}

fn evaluate_hand(cards: &[Card]) -> HandRank {
    let mut best_rank = HandRank::HighCard(Rank::Two); // Lowest possible hand
    for combo in cards.iter().combinations(5) {
        let rank = evaluate_five_card_hand(&combo);
        if rank > best_rank {
            best_rank = rank;
        }
    }
    best_rank
}

fn simulate_game(num_players: usize, hand_rank_counts: &mut HashMap<&'static str, usize>) -> usize {
    let mut deck = Deck::new();
    deck.shuffle();

    // Deal two hole cards to each player
    let mut players: Vec<Player> = (0..num_players)
        .map(|_| Player {
            hand: vec![deck.deal().unwrap(), deck.deal().unwrap()],
        })
        .collect();

    // Shuffle the players to randomize their order
    players.shuffle(&mut thread_rng());

    // Deal five community cards
    let mut community_cards = Vec::with_capacity(5);
    for _ in 0..5 {
        community_cards.push(deck.deal().unwrap());
    }

    // Evaluate each player's best hand
    let mut best_hand_rank = HandRank::HighCard(Rank::Two);
    let mut winner_indices = vec![];

    for (i, player) in players.iter().enumerate() {
        let mut all_cards = player.hand.clone();
        all_cards.extend_from_slice(&community_cards);
        let hand_rank = evaluate_hand(&all_cards);

        // Get the category of the hand rank
        let category = hand_rank_category(&hand_rank);

        // Count the hand rank category locally
        *hand_rank_counts.entry(category).or_insert(0) += 1;

        if hand_rank > best_hand_rank {
            best_hand_rank = hand_rank;
            winner_indices.clear();
            winner_indices.push(i);
        } else if hand_rank == best_hand_rank {
            winner_indices.push(i);
        }
    }

    // Randomly select a winner among tied players
    let mut rng = thread_rng();
    let winner = *winner_indices.choose(&mut rng).unwrap();

    winner
}

fn main() {
    let num_games = 1_000_000;
    let num_players = 6;

    // Use Arc and Mutex for shared data
    let wins = Arc::new(Mutex::new(vec![0usize; num_players]));
    let hand_rank_counts = Arc::new(Mutex::new(HashMap::new()));

    (0..num_games).into_par_iter().for_each(|_| {
        let mut local_hand_rank_counts: HashMap<&'static str, usize> = HashMap::new();

        let winner = simulate_game(num_players, &mut local_hand_rank_counts);

        // Update wins
        {
            let mut wins_lock = wins.lock().unwrap();
            wins_lock[winner] += 1;
        }

        // Update hand rank counts
        {
            let mut hand_rank_counts_lock = hand_rank_counts.lock().unwrap();
            for (key, count) in local_hand_rank_counts {
                *hand_rank_counts_lock.entry(key).or_insert(0) += count;
            }
        }
    });

    // Retrieve the results
    let wins = Arc::try_unwrap(wins).unwrap().into_inner().unwrap();
    let hand_rank_counts = Arc::try_unwrap(hand_rank_counts)
        .unwrap()
        .into_inner()
        .unwrap();

    // Display player wins
    for (i, &win_count) in wins.iter().enumerate() {
        println!("Player {} wins {} times", i + 1, win_count);
    }

    // Display most common hand rank categories
    println!("\nHand rank frequencies:");
    let mut hand_rank_vec: Vec<(&&'static str, &usize)> = hand_rank_counts.iter().collect();
    hand_rank_vec.sort_by(|a, b| b.1.cmp(a.1)); // Sort by frequency descending

    let total_hands = num_games * num_players;

    for (hand_rank, count) in hand_rank_vec {
        let percentage = (*count as f64 / total_hands as f64) * 100.0;
        println!("{}: {} times ({:.4}%)", hand_rank, count, percentage);
    }
}
