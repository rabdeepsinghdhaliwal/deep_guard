# OpenTimestamps example proof

`hello-world.txt` and `hello-world.txt.ots` are the example timestamp shipped
with the official OpenTimestamps client
(https://github.com/opentimestamps/opentimestamps-client/tree/master/examples,
LGPL-3.0). The proof says the file existed by Bitcoin block 358391, mined
2015-05-28 15:41:18 UTC.

The tests check Deep-Guard's own verifier against it offline. The block's
Merkle root they compare with was looked up on blockstream.info and
blockcypher.com on 2026-11-04; both gave
`8a1b66ecb7cbd07d8139a7e7d7f2c41aab1f5009b8364aaf61d03ad245e47e00`.
