# Rule integrity and v2 integration audit

The v2 policy branch was cut from v9, whereas main/v12 added task recovery, retention and quality analytics. The integration reconciles both.

## Rules and regression

Rank order is 3, 4, ... Q, K, A, 2, small joker, big joker. An ordinary follow must have the same type, same length and greater main rank. Pair Q and pair A cannot beat pair 2. A normal four-of-a-kind bomb or joker rocket can beat an ordinary pair. After two successive PASS actions, the trick resets: pair Q can start a new trick but does not beat the earlier pair 2.

The authoritative GameEngine generates legal candidates and validates exact membership before state mutation. DDZEnv.step consumes only indices into the same legal list. Models, encoders, self-play and Arena are not permitted to bypass this rule engine. Regression tests verify pair 2 vs pair Q, bomb exceptions, candidate legality, three seeded full episodes, and immutable state on rejected actions.

## Debug display

Enable locally with ddz-arena --host 127.0.0.1 --allow-baseline --allow-debug-hands. The separate, no-store debug endpoint shows all three sorted hands, but the normal snapshot and WebSocket remain redacted. The debug data is never fed into agent observations or evaluation.

## Training and compatibility

Feature branch introduces a v2 state/action encoder, schema-aware network checkpoint loading, different bid/play exploration, reasonable baseline and redeal penalty. v12 checkpoint retention and Arena journal recovery are preserved. A v1 weight file cannot be renamed into v2 schema; changing bidding credit assignment requires a fresh experiment.

The tests establish rules integrity and integration behavior, not strong playing ability.
