import functions as func

import secrets
import tenseal as ts
from phe import paillier
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

k_value = 123.325
random_coef = secrets.randbelow(10 - 3 + 1) + 3

batch_size = 4
learning_rate = 0.01
regularization_rate = 0.001

nn_input_shape = 0
n_parties = 2
n_servers = 2

# poly_mod_degree = 16384
# coeff_mod_bit_sizes = [60, 40, 60, 40, 60]
poly_mod_degree = 2048
coeff_mod_bit_sizes = [20, 20]

context = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=poly_mod_degree,
                     coeff_mod_bit_sizes=coeff_mod_bit_sizes)
context.global_scale = global_scale = 2 ** 10
context.generate_galois_keys()

key_size = 256
public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)

type_HE = False
type_paillier = False
type_DP = False

# key_FE = RSA.generate(512)
# public_key_FE = key_FE.publickey()
# encryptor = PKCS1_OAEP.new(public_key_FE)
# decryptor = PKCS1_OAEP.new(key_FE)

private_key_fe = ec.generate_private_key(ec.SECP256R1(), default_backend())
public_key_fe = private_key_fe.public_key()
shared_key = func.derive_shared_key(private_key_fe, public_key_fe)
