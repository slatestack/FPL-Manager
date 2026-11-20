import pytest

from app.ingestion.names import PlayerResolver, TeamResolver, normalize

TEAMS = [
    {"team_id": 1, "name": "Arsenal", "short_name": "ARS"},
    {"team_id": 2, "name": "Man City", "short_name": "MCI"},
    {"team_id": 3, "name": "Spurs", "short_name": "TOT"},
    {"team_id": 4, "name": "Wolves", "short_name": "WOL"},
    {"team_id": 5, "name": "Man Utd", "short_name": "MUN"},
    {"team_id": 6, "name": "Nott'm Forest", "short_name": "NFO"},
    {"team_id": 7, "name": "Brighton", "short_name": "BHA"},
]

PLAYERS = [
    {"player_id": 2, "team_id": 1, "web_name": "Gabriel", "first_name": "Gabriel", "second_name": "dos Santos Magalhães"},
    {"player_id": 3, "team_id": 1, "web_name": "Saka", "first_name": "Bukayo", "second_name": "Saka"},
    {"player_id": 10, "team_id": 1, "web_name": "Ødegaard", "first_name": "Martin", "second_name": "Ødegaard"},
    {"player_id": 5, "team_id": 2, "web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland"},
    {"player_id": 8, "team_id": 3, "web_name": "Son", "first_name": "Heung-Min", "second_name": "Son"},
    {"player_id": 20, "team_id": 5, "web_name": "B.Fernandes", "first_name": "Bruno Miguel", "second_name": "Borges Fernandes"},
]


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Martin Ødegaard") == "martin odegaard"
    assert normalize("Nott'm Forest") == "nottm forest"
    assert normalize("Brighton & Hove Albion") == "brighton and hove albion"


@pytest.mark.parametrize(
    ("external", "team_id"),
    [
        ("Arsenal", 1),
        ("Manchester City", 2),
        ("Tottenham Hotspur", 3),
        ("Tottenham", 3),
        ("Wolverhampton Wanderers", 4),
        ("Manchester United", 5),
        ("Nottingham Forest", 6),
        ("Brighton and Hove Albion", 7),
        ("MCI", 2),
        ("Liverpool", None),
    ],
)
def test_team_resolution(external, team_id):
    assert TeamResolver(TEAMS).resolve(external) == team_id


@pytest.mark.parametrize(
    ("name", "teams", "player_id"),
    [
        ("Erling Haaland", [2], 5),
        ("Bukayo Saka", [1], 3),
        ("Martin Odegaard", [1], 10),
        ("Gabriel Magalhães", [1], 2),
        ("Son Heung-Min", [3], 8),
        ("Bruno Fernandes", [5], 20),
        ("Haaland", [], 5),
        ("Some Loanee", [4], None),
    ],
)
def test_player_resolution(name, teams, player_id):
    match = PlayerResolver(PLAYERS).resolve(name, teams)
    assert (match.player_id if match else None) == player_id


def test_exact_match_flagged():
    match = PlayerResolver(PLAYERS).resolve("Bukayo Saka", [1])
    assert match is not None and match.matched_by == "exact" and match.confidence == 1.0


def test_ambiguous_match_is_rejected():
    twins = [
        {"player_id": 1, "team_id": 1, "web_name": "J.Smith", "first_name": "John", "second_name": "Smith"},
        {"player_id": 2, "team_id": 1, "web_name": "J.Smith", "first_name": "Jon", "second_name": "Smith"},
    ]
    assert PlayerResolver(twins).resolve("J Smith", [1]) is None
