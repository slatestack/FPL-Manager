-- Phase 2: tables populated by the ingestion & enrichment pipeline.
-- Section 8 tables are left untouched apart from additive columns on `players`.

-- ---------------------------------------------------------------- reference data
CREATE TABLE teams (
    team_id                INT PRIMARY KEY,           -- FPL team id (1..20, changes each season)
    code                   INT NOT NULL,              -- stable cross-season club code
    name                   VARCHAR(64) NOT NULL,
    short_name             VARCHAR(8) NOT NULL,
    strength               INT,
    strength_overall_home  INT,
    strength_overall_away  INT,
    strength_attack_home   INT,
    strength_attack_away   INT,
    strength_defence_home  INT,
    strength_defence_away  INT,
    updated_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE gameweeks (
    gameweek_id        INT PRIMARY KEY,
    name               VARCHAR(32) NOT NULL,
    deadline_time      TIMESTAMPTZ NOT NULL,
    is_previous        BOOLEAN NOT NULL DEFAULT FALSE,
    is_current         BOOLEAN NOT NULL DEFAULT FALSE,
    is_next            BOOLEAN NOT NULL DEFAULT FALSE,
    finished           BOOLEAN NOT NULL DEFAULT FALSE,
    data_checked       BOOLEAN NOT NULL DEFAULT FALSE,
    average_entry_score INT,
    highest_score      INT,
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX gameweeks_deadline_idx ON gameweeks (deadline_time);

CREATE TABLE chip_definitions (
    chip_id      INT PRIMARY KEY,
    name         VARCHAR(16) NOT NULL,   -- 'wildcard', 'freehit', 'bboost', '3xc'
    chip_type    VARCHAR(16),
    start_event  INT,
    stop_event   INT,
    number       INT,
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE fixtures (
    fixture_id         INT PRIMARY KEY,
    code               INT,
    gameweek_id        INT,                       -- NULL while a fixture is postponed/unscheduled
    team_h             INT NOT NULL REFERENCES teams(team_id),
    team_a             INT NOT NULL REFERENCES teams(team_id),
    team_h_difficulty  INT,
    team_a_difficulty  INT,
    kickoff_time       TIMESTAMPTZ,
    started            BOOLEAN,
    finished           BOOLEAN NOT NULL DEFAULT FALSE,
    finished_provisional BOOLEAN NOT NULL DEFAULT FALSE,
    team_h_score       INT,
    team_a_score       INT,
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX fixtures_gameweek_idx ON fixtures (gameweek_id);
CREATE INDEX fixtures_team_h_idx ON fixtures (team_h);
CREATE INDEX fixtures_team_a_idx ON fixtures (team_a);

-- ---------------------------------------------------------------- players (additive)
ALTER TABLE players
    ADD COLUMN code                         INT,
    ADD COLUMN first_name                   VARCHAR(64),
    ADD COLUMN second_name                  VARCHAR(64),
    ADD COLUMN news                         TEXT,
    ADD COLUMN news_added                   TIMESTAMPTZ,
    ADD COLUMN chance_of_playing_this_round INT,
    ADD COLUMN selected_by_percent          NUMERIC(5,2),
    ADD COLUMN form                         NUMERIC(5,2),
    ADD COLUMN points_per_game              NUMERIC(5,2),
    ADD COLUMN ep_next                      NUMERIC(5,2),
    ADD COLUMN total_points                 INT,
    ADD COLUMN minutes                      INT,
    ADD COLUMN starts                       INT,
    ADD COLUMN expected_goals               NUMERIC(6,2),
    ADD COLUMN expected_assists             NUMERIC(6,2),
    ADD COLUMN expected_goals_conceded      NUMERIC(6,2),
    ADD COLUMN penalties_order              INT,
    ADD COLUMN corners_and_indirect_freekicks_order INT,
    ADD COLUMN direct_freekicks_order       INT,
    ADD COLUMN cost_change_event            INT,
    ADD COLUMN transfers_in_event           INT,
    ADD COLUMN transfers_out_event          INT;

ALTER TABLE players
    ADD CONSTRAINT players_team_fk FOREIGN KEY (team_id) REFERENCES teams(team_id) DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX players_team_idx ON players (team_id);

-- Price and availability changes are captured by trigger so they cannot be missed
-- regardless of which code path writes `players`.
CREATE TABLE player_price_history (
    id          BIGSERIAL PRIMARY KEY,
    player_id   INT NOT NULL REFERENCES players(player_id),
    old_cost    INT,
    new_cost    INT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX player_price_history_player_idx ON player_price_history (player_id, observed_at);

CREATE TABLE player_status_history (
    id                 BIGSERIAL PRIMARY KEY,
    player_id          INT NOT NULL REFERENCES players(player_id),
    status             VARCHAR(4) NOT NULL,
    chance_of_playing  INT,
    news               TEXT,
    observed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX player_status_history_player_idx ON player_status_history (player_id, observed_at);

CREATE FUNCTION record_player_changes() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR NEW.now_cost IS DISTINCT FROM OLD.now_cost THEN
        INSERT INTO player_price_history (player_id, old_cost, new_cost)
        VALUES (NEW.player_id, CASE WHEN TG_OP = 'UPDATE' THEN OLD.now_cost END, NEW.now_cost);
    END IF;
    IF TG_OP = 'INSERT'
       OR NEW.status IS DISTINCT FROM OLD.status
       OR NEW.chance_of_playing IS DISTINCT FROM OLD.chance_of_playing
       OR NEW.news IS DISTINCT FROM OLD.news THEN
        INSERT INTO player_status_history (player_id, status, chance_of_playing, news)
        VALUES (NEW.player_id, NEW.status, NEW.chance_of_playing, NEW.news);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER players_record_changes
    AFTER INSERT OR UPDATE ON players
    FOR EACH ROW EXECUTE FUNCTION record_player_changes();

-- ---------------------------------------------------------------- per-match history (element-summary)
CREATE TABLE player_gw_history (
    player_id                  INT NOT NULL REFERENCES players(player_id),
    fixture_id                 INT NOT NULL,
    gameweek_id                INT NOT NULL,
    opponent_team              INT NOT NULL,
    was_home                   BOOLEAN NOT NULL,
    kickoff_time               TIMESTAMPTZ,
    minutes                    INT NOT NULL,
    starts                     INT,
    total_points               INT NOT NULL,
    goals_scored               INT NOT NULL,
    assists                    INT NOT NULL,
    clean_sheets               INT NOT NULL,
    goals_conceded             INT NOT NULL,
    own_goals                  INT NOT NULL DEFAULT 0,
    penalties_saved            INT NOT NULL DEFAULT 0,
    penalties_missed           INT NOT NULL DEFAULT 0,
    yellow_cards               INT NOT NULL DEFAULT 0,
    red_cards                  INT NOT NULL DEFAULT 0,
    saves                      INT NOT NULL DEFAULT 0,
    bonus                      INT NOT NULL DEFAULT 0,
    bps                        INT NOT NULL DEFAULT 0,
    defensive_contribution     INT,
    expected_goals             NUMERIC(6,2),
    expected_assists           NUMERIC(6,2),
    expected_goal_involvements NUMERIC(6,2),
    expected_goals_conceded    NUMERIC(6,2),
    value                      INT,
    selected                   INT,
    updated_at                 TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, fixture_id)
);
CREATE INDEX player_gw_history_gw_idx ON player_gw_history (gameweek_id);

-- Rolling last-5-appearances per-90 form, used as features by the Phase 3 xP engine.
CREATE VIEW player_rolling_form AS
WITH ranked AS (
    SELECT h.*,
           ROW_NUMBER() OVER (PARTITION BY h.player_id ORDER BY h.kickoff_time DESC, h.fixture_id DESC) AS rn
    FROM player_gw_history h
)
SELECT player_id,
       COUNT(*)                                       AS matches,
       SUM(minutes)                                   AS minutes,
       AVG(minutes)::NUMERIC(6,2)                     AS avg_minutes,
       SUM(CASE WHEN minutes >= 60 THEN 1 ELSE 0 END) AS sixty_plus_apps,
       SUM(expected_goals)                            AS xg,
       SUM(expected_assists)                          AS xa,
       CASE WHEN SUM(minutes) > 0 THEN (SUM(expected_goals)   * 90.0 / SUM(minutes))::NUMERIC(6,3) END AS xg_per90,
       CASE WHEN SUM(minutes) > 0 THEN (SUM(expected_assists) * 90.0 / SUM(minutes))::NUMERIC(6,3) END AS xa_per90,
       AVG(bps)::NUMERIC(6,2)                         AS avg_bps,
       AVG(bonus)::NUMERIC(6,2)                       AS avg_bonus
FROM ranked
WHERE rn <= 5
GROUP BY player_id;

-- ---------------------------------------------------------------- manager profile (public entry endpoint)
CREATE TABLE manager_entries (
    entry_id                    INT PRIMARY KEY,
    entry_name                  VARCHAR(128),
    player_first_name           VARCHAR(64),
    player_last_name            VARCHAR(64),
    current_event               INT,
    summary_overall_points      INT,
    summary_overall_rank        INT,
    summary_event_points        INT,
    last_deadline_bank          INT,
    last_deadline_value         INT,
    last_deadline_total_transfers INT,
    updated_at                  TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------- enrichment: xG / xA
CREATE TABLE player_id_map (
    source       VARCHAR(32) NOT NULL,          -- e.g. 'understat'
    external_id  VARCHAR(64) NOT NULL,
    player_id    INT NOT NULL REFERENCES players(player_id),
    external_name VARCHAR(128),
    confidence   NUMERIC(4,3) NOT NULL,
    matched_by   VARCHAR(16) NOT NULL CHECK (matched_by IN ('exact', 'fuzzy', 'manual')),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source, external_id)
);

CREATE TABLE player_xg_stats (
    source        VARCHAR(32) NOT NULL,
    season        INT NOT NULL,                  -- season start year, e.g. 2026 for 2026/27
    external_id   VARCHAR(64) NOT NULL,
    player_id     INT REFERENCES players(player_id),  -- NULL when no confident FPL match
    external_name VARCHAR(128) NOT NULL,
    team_name     VARCHAR(128),
    games         INT,
    minutes       INT,
    goals         INT,
    npg           INT,
    assists       INT,
    shots         INT,
    key_passes    INT,
    xg            NUMERIC(7,3),
    npxg          NUMERIC(7,3),
    xa            NUMERIC(7,3),
    xg_chain      NUMERIC(7,3),
    xg_buildup    NUMERIC(7,3),
    npxg_per90    NUMERIC(6,3),
    xa_per90      NUMERIC(6,3),
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source, season, external_id)
);
CREATE INDEX player_xg_stats_player_idx ON player_xg_stats (player_id);

-- ---------------------------------------------------------------- enrichment: bookmaker odds
-- Append-only snapshots so each optimisation run can reference the exact odds it used.
CREATE TABLE fixture_odds (
    fixture_id      INT NOT NULL REFERENCES fixtures(fixture_id),
    source          VARCHAR(32) NOT NULL,
    captured_at     TIMESTAMPTZ NOT NULL,
    bookmaker_count INT NOT NULL,
    p_home          NUMERIC(5,4) NOT NULL,
    p_draw          NUMERIC(5,4) NOT NULL,
    p_away          NUMERIC(5,4) NOT NULL,
    p_over_2_5      NUMERIC(5,4),
    lambda_home     NUMERIC(5,3) NOT NULL,       -- implied expected goals (Poisson)
    lambda_away     NUMERIC(5,3) NOT NULL,
    p_cs_home       NUMERIC(5,4) NOT NULL,       -- P(home team keeps a clean sheet)
    p_cs_away       NUMERIC(5,4) NOT NULL,
    PRIMARY KEY (fixture_id, source, captured_at)
);

CREATE TABLE player_goal_odds (
    player_id       INT NOT NULL REFERENCES players(player_id),
    fixture_id      INT NOT NULL REFERENCES fixtures(fixture_id),
    source          VARCHAR(32) NOT NULL,
    captured_at     TIMESTAMPTZ NOT NULL,
    bookmaker_count INT NOT NULL,
    median_decimal_odds NUMERIC(7,2) NOT NULL,
    p_anytime_goal  NUMERIC(5,4) NOT NULL,
    PRIMARY KEY (player_id, fixture_id, source, captured_at)
);

-- Latest odds snapshot per fixture (convenience for downstream consumers).
CREATE VIEW latest_fixture_odds AS
SELECT DISTINCT ON (fixture_id) *
FROM fixture_odds
ORDER BY fixture_id, captured_at DESC;

-- ---------------------------------------------------------------- audit of ingestion
CREATE TABLE ingestion_runs (
    id             BIGSERIAL PRIMARY KEY,
    job            VARCHAR(32) NOT NULL,
    status         VARCHAR(16) NOT NULL CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED', 'SKIPPED')),
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ,
    rows_written   INT,
    payload_hash   VARCHAR(64),
    details        JSONB,
    error_message  TEXT
);
CREATE INDEX ingestion_runs_job_idx ON ingestion_runs (job, started_at DESC);

-- Raw upstream payloads, deduplicated by content hash, for reproducibility of any run.
CREATE TABLE raw_snapshots (
    source        VARCHAR(64) NOT NULL,
    payload_hash  VARCHAR(64) NOT NULL,
    payload       JSONB NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, payload_hash)
);
