# Kemeleon Python Implementation

> [!CAUTION]
> This is a reference implementation and has not been designed to be secure
> against any form of side-channel attack. 

This repository contains a python implementation of Kemeleon encodings, as 
specified in [draft-irtf-cfrg-kemeleon](https://datatracker.ietf.org/doc/draft-irtf-cfrg-kemeleon/).

<!-- ## Licenses -->

<!-- This project is dual-licensed under your choice of the MIT License OR the Apache License 2.0. -->

## Disclaimer.

This code is not constant time, or written to be performant. It was
written so that the python code follows the Kemeleon Encodings 
[Internet-Draft](https://datatracker.ietf.org/doc/draft-irtf-cfrg-kemeleon/)
and to be compatible with [kyber-py](https://github.com/GiacomoPope/kyber-py).
No cryptographic guarantees are made of this work.

## Using kemeleon-py

This package isn't published on PyPI; install it directly from GitHub:
```
pip install "kemeleon-py @ git+https://github.com/ssveitch/kemeleon-py.git"
```

This pulls in [pycryptodome](https://pypi.org/project/pycryptodome/) as a
dependency.

To run the test suite (which requires kyber-py),
install the dev extra instead:
```
pip install "kemeleon-py[dev] @ git+https://github.com/ssveitch/kemeleon-py.git"
```

#### Example

The main variant always succeeds. Each call to `encode_ek` / `encode_ctxt` returns
a fixed-length bytestring that is indistinguishable from random.

```python
from kyber_py.ml_kem import ML_KEM_768
from kemeleon import Kemeleon_768

ek, dk = ML_KEM_768.keygen()
K, c = ML_KEM_768.encaps(ek)

# Encode the encapsulation key and ciphertext
eek = Kemeleon_768.encode_ek(ek)
ec  = Kemeleon_768.encode_ctxt(c)

# Decode back to original ML-KEM format
assert Kemeleon_768.decode_ek(eek)   == ek
assert Kemeleon_768.decode_ctxt(ec)  == c
```

The rejection-sampling variant (`KemeleonR`, Section 5.1) produces shorter
output but may return `None` for a given input. In that case, generate a 
fresh key or ciphertext and retry.

```python
from kyber_py.ml_kem import ML_KEM_768
from kemeleon import KemeleonR_768

# Encoding key with rejection sampling
eek = None
while eek is None:
    ek, dk = ML_KEM_768.keygen()
    eek = KemeleonR_768.encode_ek(ek)
assert KemeleonR_768.decode_ek(eek) == ek

# Encoding ciphertext with rejection sampling
ec = None
while ec is None:
    K, c = ML_KEM_768.encaps(ek)
    ec = KemeleonR_768.encode_ctxt(c)
assert KemeleonR_768.decode_ctxt(ec) == c
```

