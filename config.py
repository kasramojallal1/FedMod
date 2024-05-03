import secrets
from phe import paillier
import tenseal as ts

k_value = 123
random_coef = secrets.randbelow(10 - 3 + 1) + 3

learning_rate = 0.001
regularization_rate = 0.001

plot_intervals = 60
n_parties = 10
n_servers = 2

# poly_modulus_degree = 16384
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
type_paillier = True
type_DP = False
