"""
Python implementation of Kemeleon encodings for ML-KEM.
https://datatracker.ietf.org/doc/draft-irtf-cfrg-kemeleon/

Reference implementation only - NOT for production use.
"""

import os
from typing import Optional

from Crypto.Hash import SHAKE256

# Constants

_Q = 3329                   # ML-KEM modulus
_N = 256                    # polynomial degree
_Q_POW_N = _Q ** _N
_D_T = 12                    # bit-width for uncompressed (mod-q) coefficients
_B = 2996                    # b = (q^n - 1).bit_length() = 2996
_VECTOR_BITS = 3072          # b + t = 3072 (384 bytes, byte-aligned), t = 76.
_VECTOR_BYTES = _VECTOR_BITS // 8  # 384


# Helpers

def _compress(x: int, d: int) -> int:
    """Compress_d: x -> round((2^d / q) * x) % 2^d"""
    t = 1 << d
    return ((t * x + 1664) // _Q) % t   # 1664 = q // 2


def _decompress(y: int, d: int) -> int:
    """Decompress_d: y -> round((q / 2^d) * x)"""
    t = 1 << (d - 1)
    return (_Q * y + t) >> d


def _new_xof(seed: Optional[bytes] = None):
    """
    Seed a SHAKE256 XOF and return it as a byte stream.
    If seed is not provided, a fresh 32-byte seed is drawn from os.urandom.
    """
    if seed is None:
        seed = os.urandom(32)
    elif len(seed) != 32:
        raise ValueError(f"seed must be 32 bytes, got {len(seed)}")
    return SHAKE256.new(seed)


def _randbelow(n: int, xof) -> int:
    """Uniform random integer in [0, n-1], drawn from the xof stream."""
    if n <= 1:
        return 0
    k = (n - 1).bit_length()
    mask = (1 << k) - 1
    while True:
        r = int.from_bytes(xof.read((k + 7) // 8), "little") & mask
        if r < n:
            return r


def _decode_poly_coeffs(data: bytes, d: int) -> list[int]:
    """Decode 32*d bytes into 256 coefficients with d bits each (ML-KEM byte format)."""
    assert len(data) == 32 * d
    m = _Q if d == _D_T else (1 << d)
    b_int = int.from_bytes(data, "little")
    mask = (1 << d) - 1
    coeffs = [0 for _ in range(_N)]
    for i in range(_N):
        coeffs[i] = (b_int & mask) % m
        b_int >>= d
    return coeffs


def _encode_poly_coeffs(coeffs: list[int], d: int) -> bytes:
    """Encode 256 coefficients into 32*d bytes with d bits each (ML-KEM byte format)."""
    assert len(coeffs) == _N
    t = 0
    for i in range(_N - 1):
        t |= coeffs[_N - 1 - i]
        t <<= d
    t |= coeffs[0]
    return t.to_bytes(32 * d, "little")


def _vector_encode(coeffs: list[int], xof) -> bytes:
    """
    VectorEncode: encode n=256 coefficients as 384 bytes.
    """
    assert len(coeffs) == _N
    r = 0
    q_pow = 1
    for c in coeffs:
        r += q_pow * c
        q_pow *= _Q
    # q_pow == q^n
    # m in [0, floor((2^3072 - r) / q^n)]
    max_m = ((1 << _VECTOR_BITS) - 1 - r) // q_pow
    m = _randbelow(max_m + 1, xof)
    r += m * q_pow
    return r.to_bytes(_VECTOR_BYTES, "little")


def _vector_decode(data: bytes) -> list[int]:
    """VectorDecode: decode 384 bytes into n=256 coefficients in [0, q-1]."""
    assert len(data) == _VECTOR_BYTES
    r = int.from_bytes(data, "little") % _Q_POW_N
    coeffs = [0 for _ in range(_N)]
    for i in range(_N):
        coeffs[i] = r % _Q
        r //= _Q
    return coeffs


def _sample_preimage(d: int, u: int, c: int, xof) -> int:
    """
    SamplePreimage: return a uniform random v in Z_q such that Compress_d(v) == c.
    """
    if d == 10:
        if _compress((u + 2) % _Q, d) == c:
            rand = _randbelow(4, xof) - 1      # {-1, 0, 1, 2}
        elif _compress((u - 2) % _Q, d) == c:
            rand = _randbelow(4, xof) - 2      # {-2, -1, 0, 1}
        else:
            rand = _randbelow(3, xof) - 1      # {-1, 0, 1}
    elif d == 11:
        if _compress((u + 1) % _Q, d) == c:
            rand = _randbelow(2, xof)          # {0, 1}
        elif _compress((u - 1) % _Q, d) == c:
            rand = _randbelow(2, xof) - 1      # {-1, 0}
        else:
            rand = 0
    elif d == 5:
        if u == 0:
            rand = _randbelow(105, xof) - 52   # {-52, …, 52}
        elif u <= 1560:
            rand = _randbelow(104, xof) - 51   # {-51, …, 52}
        else:
            rand = _randbelow(104, xof) - 52   # {-52, …, 51}
    elif d == 4:
        if u == 0:
            rand = _randbelow(209, xof) - 104  # {-104, …, 104}
        elif u <= 1456:
            rand = _randbelow(208, xof) - 103  # {-103, …, 104}
        else:
            rand = _randbelow(208, xof) - 104  # {-104, …, 103}
    else:
        raise ValueError(f"Unsupported compression parameter d={d}")
    return (u + rand) % _Q


# (rejection-sampling variant)

def _vector_encode_r(
    flat_coeffs: list[int], num_bits: int, num_bytes: int, xof
) -> Optional[bytes]:
    """
    VectorEncodeR: encode k*n coefficients into ceil(k*n*log2(q)/8) bytes.

    Returns None (rejection) when the MSB is 1.
    Unused high bits in the top byte are randomized.
    """
    r = 0
    q_pow = 1
    for c in flat_coeffs:
        r += q_pow * c
        q_pow *= _Q

    if (r >> (num_bits - 1)) & 1:
        return None

    unused = num_bytes * 8 - num_bits + 1
    r |= _randbelow(1 << unused, xof) << (num_bits - 1)

    return r.to_bytes(num_bytes, "little")


def _vector_decode_r(data: bytes, k: int, num_bits: int, num_bytes: int) -> list[int]:
    """VectorDecodeR: decode ceil(k*n*log2(q)/8) bytes into k*n coefficients."""
    k_n = k * _N
    assert len(data) == num_bytes

    r = int.from_bytes(data, "little") & ((1 << (num_bits - 1)) - 1)
    coeffs = [0 for _ in range(k_n)]
    for i in range(k_n):
        coeffs[i] = r % _Q
        r //= _Q
    return coeffs


class Kemeleon:
    """
    Kemeleon encodings for a specific ML-KEM parameter set (Section 4).

    encode_ek / decode_ek     — encapsulation key (always succeeds)
    encode_ctxt / decode_ctxt — ciphertext (always succeeds)
    """

    def __init__(self, params: dict):
        self.k = params["k"]
        self.du = params["du"]
        self.dv = params["dv"]

        self._ek_size = 384 * self.k + 32
        self._c1_size = 32 * self.du * self.k
        self._c2_size = 32 * self.dv
        self._c_size = self._c1_size + self._c2_size
        self._eek_size = _VECTOR_BYTES * self.k + 32
        self._ec_size = _VECTOR_BYTES * (self.k + 1)

    def eek_size(self) -> int:
        """Encoded encapsulation key size."""
        return self._eek_size

    def ec_size(self) -> int:
        """Encoded ciphertext size."""
        return self._ec_size

    def encode_ek(self, ek: bytes) -> bytes:
        """
        EncodeEk: encode ML-KEM encapsulation key as a uniform bytestring.
        """
        if len(ek) != self._ek_size:
            raise ValueError(
                f"ek must be {self._ek_size} bytes, got {len(ek)}"
            )
        t_bytes = ek[: 384 * self.k]
        rho = ek[384 * self.k :]

        xof = _new_xof()
        parts = [b""] * self.k
        for i in range(self.k):
            coeffs = _decode_poly_coeffs(t_bytes[384 * i : 384 * (i + 1)], _D_T)
            parts[i] = _vector_encode(coeffs, xof)
        return b"".join(parts) + rho

    def decode_ek(self, eek: bytes) -> bytes:
        """
        DecodeEk: decode Kemeleon-encoded encapsulation key to ML-KEM format.
        """
        expected = self.eek_size()
        if len(eek) != expected:
            raise ValueError(
                f"eek must be {expected} bytes, got {len(eek)}"
            )
        rho = eek[_VECTOR_BYTES * self.k :]
        t_bytes = b""
        for i in range(self.k):
            coeffs = _vector_decode(eek[_VECTOR_BYTES * i : _VECTOR_BYTES * (i + 1)])
            t_bytes += _encode_poly_coeffs(coeffs, _D_T)
        return t_bytes + rho

    def encode_ctxt(self, c: bytes) -> bytes:
        """
        EncodeCtxt: encode ML-KEM ciphertext as a uniform bytestring.
        """
        if len(c) != self._c_size:
            raise ValueError(
                f"c must be {self._c_size} bytes, got {len(c)}"
            )
        c1 = c[: self._c1_size]
        c2 = c[self._c1_size :]

        xof = _new_xof()
        parts = [b""] * (self.k + 1)
        for i in range(self.k):
            comp = _decode_poly_coeffs(
                c1[32 * self.du * i : 32 * self.du * (i + 1)], self.du
            )
            preimage = [
                _sample_preimage(self.du, _decompress(cc, self.du), cc, xof)
                for cc in comp
            ]
            parts[i] = _vector_encode(preimage, xof)

        comp_v = _decode_poly_coeffs(c2, self.dv)
        preimage_v = [
            _sample_preimage(self.dv, _decompress(cc, self.dv), cc, xof)
            for cc in comp_v
        ]
        parts[self.k] = _vector_encode(preimage_v, xof)
        return b"".join(parts)

    def decode_ctxt(self, ec: bytes) -> bytes:
        """
        DecodeCtxt: decode Kemeleon-encoded ciphertext to ML-KEM format.
        """
        expected = self.ec_size()
        if len(ec) != expected:
            raise ValueError(
                f"ec must be {expected} bytes, got {len(ec)}"
            )
        c1_parts = [b""] * self.k
        for i in range(self.k):
            coeffs = _vector_decode(ec[_VECTOR_BYTES * i : _VECTOR_BYTES * (i + 1)])
            compressed = [_compress(x, self.du) for x in coeffs]
            c1_parts[i] = _encode_poly_coeffs(compressed, self.du)

        v_coeffs = _vector_decode(
            ec[_VECTOR_BYTES * self.k : _VECTOR_BYTES * (self.k + 1)]
        )
        compressed_v = [_compress(x, self.dv) for x in v_coeffs]
        c2 = _encode_poly_coeffs(compressed_v, self.dv)
        return b"".join(c1_parts) + c2


class KemeleonR:
    """
    Kemeleon encodings with rejection sampling for smaller outputs
    (Section 5.1)

    encode_ek / decode_ek     — encapsulation key
    encode_ctxt / decode_ctxt — ciphertext
    """

    def __init__(self, params: dict):
        self.k = params["k"]
        self.du = params["du"]
        self.dv = params["dv"]

        self._ek_size = 384 * self.k + 32
        self._c1_size = 32 * self.du * self.k
        self._c2_size = 32 * self.dv
        self._c_size = self._c1_size + self._c2_size

        k_n = self.k * _N
        # Bit/byte count for the k*n-coefficient VectorEncodeR encoding.
        self._r_bits = (_Q ** k_n - 1).bit_length()
        self._r_bytes = (self._r_bits + 7) // 8
        self._eek_size = self._r_bytes + 32
        self._ec_size = self._r_bytes + self._c2_size

    def eek_size(self) -> int:
        """Encoded encapsulation key size."""
        return self._eek_size

    def ec_size(self) -> int:
        """Encoded ciphertext size."""
        return self._ec_size

    def encode_ek(self, ek: bytes) -> Optional[bytes]:
        """
        EncodeEkR: encode ML-KEM encapsulation key (rejection-sampling variant).

        Returns None on rejection; the caller must retry.
        """
        if len(ek) != self._ek_size:
            raise ValueError(
                f"ek must be {self._ek_size} bytes, got {len(ek)}"
            )
        t_bytes = ek[: 384 * self.k]
        rho = ek[384 * self.k :]

        flat: list[int] = [0] * (self.k * _N)
        for i in range(self.k):
            flat[_N * i : _N * (i + 1)] = _decode_poly_coeffs(
                t_bytes[384 * i : 384 * (i + 1)], _D_T
            )

        xof = _new_xof()
        encoded = _vector_encode_r(flat, self._r_bits, self._r_bytes, xof)
        if encoded is None:
            return None
        return encoded + rho

    def decode_ek(self, eek: bytes) -> bytes:
        """
        DecodeEkR: decode rejection-sampled encapsulation key to ML-KEM format.
        """
        expected = self.eek_size()
        if len(eek) != expected:
            raise ValueError(
                f"eek must be {expected} bytes, got {len(eek)}"
            )
        flat = _vector_decode_r(eek[: self._r_bytes], self.k, self._r_bits, self._r_bytes)
        rho = eek[self._r_bytes :]

        t_bytes = b""
        for i in range(self.k):
            t_bytes += _encode_poly_coeffs(flat[_N * i : _N * (i + 1)], _D_T)
        return t_bytes + rho

    def encode_ctxt(self, c: bytes) -> Optional[bytes]:
        """
        EncodeCtxtR: encode ML-KEM ciphertext (rejection-sampling variant).

        Returns None on rejection; the caller must retry.
        """
        if len(c) != self._c_size:
            raise ValueError(
                f"c must be {self._c_size} bytes, got {len(c)}"
            )
        c1 = c[: self._c1_size]
        c2 = c[self._c1_size :]

        xof = _new_xof()
        flat_u: list[int] = [0] * (self.k * _N)
        for i in range(self.k):
            comp = _decode_poly_coeffs(
                c1[32 * self.du * i : 32 * self.du * (i + 1)], self.du
            )
            flat_u[_N * i : _N * (i + 1)] = (
                _sample_preimage(self.du, _decompress(cc, self.du), cc, xof)
                for cc in comp
            )

        encoded = _vector_encode_r(flat_u, self._r_bits, self._r_bytes, xof)
        if encoded is None:
            return None

        comp_v = _decode_poly_coeffs(c2, self.dv)
        reject_denom = -(-_Q // (1 << self.dv))  # ceil(q / 2^dv)
        for cc in comp_v:
            if cc == 0 and _randbelow(reject_denom, xof) == 0:
                return None

        return encoded + c2

    def decode_ctxt(self, ec: bytes) -> bytes:
        """
        DecodeCtxtR: decode ciphertext to ML-KEM format.
        """
        expected = self.ec_size()
        if len(ec) != expected:
            raise ValueError(
                f"ec must be {expected} bytes, got {len(ec)}"
            )
        flat_u = _vector_decode_r(ec[: self._r_bytes], self.k, self._r_bits, self._r_bytes)
        c2 = ec[self._r_bytes :]

        c1_parts = [b""] * self.k
        for i in range(self.k):
            compressed = [_compress(x, self.du) for x in flat_u[_N * i : _N * (i + 1)]]
            c1_parts[i] = _encode_poly_coeffs(compressed, self.du)
        return b"".join(c1_parts) + c2


Kemeleon_512 = Kemeleon({"k": 2, "du": 10, "dv": 4})
Kemeleon_768 = Kemeleon({"k": 3, "du": 10, "dv": 4})
Kemeleon_1024 = Kemeleon({"k": 4, "du": 11, "dv": 5})

KemeleonR_512 = KemeleonR({"k": 2, "du": 10, "dv": 4})
KemeleonR_768 = KemeleonR({"k": 3, "du": 10, "dv": 4})
KemeleonR_1024 = KemeleonR({"k": 4, "du": 11, "dv": 5})
