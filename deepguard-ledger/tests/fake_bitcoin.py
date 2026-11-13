"""A stand-in for the OpenTimestamps calendars and the block explorers, so the
tests never touch the network."""
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpPrepend, OpSHA256
from opentimestamps.core.timestamp import Timestamp


class FakeBitcoin:
    """Calendars that promise, then (on upgrade, once `mined`) deliver a path
    into fake Bitcoin block 900000 whose Merkle root is whatever the path ends
    on, as an explorer would report it."""
    HEIGHT = 900000

    def __init__(self):
        self.roots = {}
        self.mined = False
        self.submitted = []

    def submit(self, url, digest):
        self.submitted.append((url, digest))
        ts = Timestamp(digest)
        node = ts.ops.add(OpPrepend(url.encode()[-6:])).ops.add(OpSHA256())
        node.attestations.add(PendingAttestation("https://alice.btc.calendar.opentimestamps.org"))
        return ts

    def fetch(self, uri, commitment):
        if not self.mined:
            return None
        ts = Timestamp(commitment)
        tip = ts.ops.add(OpAppend(b"tx-bytes")).ops.add(OpSHA256()).ops.add(OpSHA256())
        tip.attestations.add(BitcoinBlockHeaderAttestation(self.HEIGHT))
        self.roots[self.HEIGHT] = tip.msg[::-1].hex()  # explorers print it reversed
        return ts

    def lookup(self, height):
        return {"height": height, "block_hash": "ab" * 32, "merkle_root": self.roots[height], "time": 1_800_000_000,
                "time_utc": "2027-01-15T08:00:00+00:00", "sources": ["fake-a", "fake-b"], "single_source": False}
