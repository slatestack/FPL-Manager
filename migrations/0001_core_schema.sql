-- Section 8 of the architecture blueprint, applied verbatim.

CREATE TABLE players (
    player_id INT PRIMARY KEY,
    web_name VARCHAR(64) NOT NULL,
    team_id INT NOT NULL,
    position VARCHAR(3) CHECK (position IN ('GKP', 'DEF', 'MID', 'FWD')),
    now_cost INT NOT NULL, -- Stored in tenths (e.g., 105 = 10.5m)
    status VARCHAR(4) NOT NULL, -- 'a', 'd', 'i', 's', 'u', 'n'
    chance_of_playing INT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE player_predictions (
    id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(player_id),
    target_gw INT NOT NULL,
    run_timestamp TIMESTAMPTZ NOT NULL,
    prob_play NUMERIC(4,3) NOT NULL,
    xp_total NUMERIC(5,2) NOT NULL,
    xp_attacking NUMERIC(5,2),
    xp_defensive NUMERIC(5,2),
    xp_bonus NUMERIC(5,2),
    UNIQUE (player_id, target_gw, run_timestamp)
);

CREATE TABLE execution_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id INT NOT NULL,
    gameweek INT NOT NULL,
    execution_type VARCHAR(16) CHECK (execution_type IN ('DRY_RUN', 'LIVE_COMMIT')),
    payload_hash VARCHAR(64) NOT NULL,
    transfers_in INT[] DEFAULT '{}',
    transfers_out INT[] DEFAULT '{}',
    starting_xi INT[] NOT NULL,
    bench_order INT[] NOT NULL,
    captain_id INT NOT NULL,
    vice_captain_id INT NOT NULL,
    chip_activated VARCHAR(16), -- NULL, 'wildcard', 'freehit', 'bboost', '3xc'
    projected_xp_gain NUMERIC(5,2),
    status VARCHAR(16) CHECK (status IN ('PENDING', 'SUCCESS', 'FAILED', 'ABORTED')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE system_config (
    team_id INT PRIMARY KEY,
    auto_execute_enabled BOOLEAN DEFAULT TRUE,
    max_hits_per_gw INT DEFAULT 1,
    chip_aggression_profile VARCHAR(16) DEFAULT 'BALANCED', -- 'CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'
    encrypted_auth_cookie TEXT,
    auth_expires_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
