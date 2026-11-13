# A real Deep-Guard anchor in Bitcoin

`block-000002.header.json` is the canonical header of block #2 of a
Deep-Guard development ledger (its SHA-256 is that block's hash,
`7923e57aa0e33013…`), and `block-000002.header.json.ots` is its
OpenTimestamps proof, made by `anchoring.py` on 2026-11-04 16:04 UTC and
upgraded on 2026-11-05.

Two calendars put it in Bitcoin, in two different blocks:

| Bitcoin block | Block hash | Merkle root (as explorers print it) | Mined (UTC) |
|---|---|---|---|
| 974545 | `00000000000000000000076e8d490afd1599c14cfb7bf3d6c3f7c520731653d8` | `01c66ef9ea807cff37f391608735a4b8fcd0c2239c3796c1cb3c37ec15413de0` | 2026-11-04 16:21:44 |
| 974547 | `00000000000000000001ca259230f213d5f48020bdbd67c13c70f5c6804a5662` | `903e6c40673c32c7921d7ea5a899dca0c852a9efa873c62def0e0611fac3062d` | 2026-11-04 16:28:44 |

Both were looked up on blockstream.info and blockcypher.com on 2026-11-05 and
agreed. `test_anchoring.py` checks this proof offline against these pinned
values, so a change to the verifier that broke real proofs fails the tests.
