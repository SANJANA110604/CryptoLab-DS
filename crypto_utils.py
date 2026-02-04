"""
Advanced Cryptography Utilities
This module provides advanced cryptographic functions and utilities
"""

import os
import hashlib
import hmac
import secrets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import base64
import json
from typing import Tuple, Optional, Dict, Any
import time
from datetime import datetime, timedelta
import re

class AdvancedCrypto:
    """Advanced cryptographic operations"""

    def __init__(self):
        self.backend = default_backend()

    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[str, str]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem.decode(), public_pem.decode()

    def rsa_encrypt(self, public_key_pem: str, plaintext: str) -> str:
        """Encrypt with RSA public key"""
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode(),
            backend=self.backend
        )

        ciphertext = public_key.encrypt(
            plaintext.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return base64.b64encode(ciphertext).decode()

    def rsa_decrypt(self, private_key_pem: str, ciphertext: str) -> str:
        """Decrypt with RSA private key"""
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=self.backend
        )

        ciphertext_bytes = base64.b64decode(ciphertext)

        plaintext = private_key.decrypt(
            ciphertext_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return plaintext.decode()

    def generate_aes_key(self, key_size: int = 256) -> str:
        """Generate AES key"""
        key = os.urandom(key_size // 8)
        return base64.b64encode(key).decode()

    def aes_encrypt(self, key: str, plaintext: str, mode: str = 'CBC') -> Tuple[str, str]:
        """Encrypt with AES"""
        key_bytes = base64.b64decode(key)

        if mode == 'CBC':
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=self.backend)
        elif mode == 'GCM':
            iv = os.urandom(12)
            cipher = Cipher(algorithms.AES(key_bytes), modes.GCM(iv), backend=self.backend)
        else:
            raise ValueError("Unsupported AES mode")

        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()

        if mode == 'GCM':
            tag = encryptor.tag
            return base64.b64encode(ciphertext).decode(), base64.b64encode(iv + tag).decode()
        else:
            return base64.b64encode(ciphertext).decode(), base64.b64encode(iv).decode()

    def aes_decrypt(self, key: str, ciphertext: str, iv: str, mode: str = 'CBC', tag: str = None) -> str:
        """Decrypt with AES"""
        key_bytes = base64.b64decode(key)
        ciphertext_bytes = base64.b64decode(ciphertext)
        iv_bytes = base64.b64decode(iv)

        if mode == 'CBC':
            cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv_bytes), backend=self.backend)
        elif mode == 'GCM':
            if tag:
                tag_bytes = base64.b64decode(tag)
                cipher = Cipher(algorithms.AES(key_bytes), modes.GCM(iv_bytes, tag_bytes), backend=self.backend)
            else:
                raise ValueError("GCM mode requires authentication tag")
        else:
            raise ValueError("Unsupported AES mode")

        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext_bytes) + decryptor.finalize()

        return plaintext.decode()

    def generate_ecdsa_keypair(self, curve: str = 'secp256r1') -> Tuple[str, str]:
        """Generate ECDSA key pair"""
        if curve == 'secp256r1':
            curve_obj = ec.SECP256R1()
        elif curve == 'secp384r1':
            curve_obj = ec.SECP384R1()
        elif curve == 'secp521r1':
            curve_obj = ec.SECP521R1()
        else:
            raise ValueError("Unsupported curve")

        private_key = ec.generate_private_key(curve_obj, self.backend)

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem.decode(), public_pem.decode()

    def ecdsa_sign(self, private_key_pem: str, message: str) -> str:
        """Sign message with ECDSA"""
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=self.backend
        )

        signature = private_key.sign(
            message.encode(),
            ec.ECDSA(hashes.SHA256())
        )

        return base64.b64encode(signature).decode()

    def ecdsa_verify(self, public_key_pem: str, message: str, signature: str) -> bool:
        """Verify ECDSA signature"""
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode(),
            backend=self.backend
        )

        signature_bytes = base64.b64decode(signature)

        try:
            public_key.verify(
                signature_bytes,
                message.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except:
            return False

class KeyDerivation:
    """Key derivation functions"""

    def pbkdf2_derive(self, password: str, salt: bytes = None, key_length: int = 32,
                     iterations: int = 100000) -> Tuple[str, str]:
        """Derive key using PBKDF2"""
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )

        key = kdf.derive(password.encode())
        return base64.b64encode(key).decode(), base64.b64encode(salt).decode()

    def scrypt_derive(self, password: str, salt: bytes = None, key_length: int = 32,
                     n: int = 32768, r: int = 8, p: int = 1) -> Tuple[str, str]:
        """Derive key using Scrypt"""
        if salt is None:
            salt = os.urandom(16)

        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            n=n, r=r, p=p,
            backend=default_backend()
        )

        key = kdf.derive(password.encode())
        return base64.b64encode(key).decode(), base64.b64encode(salt).decode()

class HashFunctions:
    """Cryptographic hash functions"""

    def sha256_hash(self, data: str) -> str:
        """Compute SHA-256 hash"""
        hash_obj = hashlib.sha256(data.encode())
        return hash_obj.hexdigest()

    def sha3_256_hash(self, data: str) -> str:
        """Compute SHA3-256 hash"""
        hash_obj = hashlib.sha3_256(data.encode())
        return hash_obj.hexdigest()

    def blake2b_hash(self, data: str, key: str = None) -> str:
        """Compute BLAKE2b hash"""
        if key:
            key_bytes = key.encode()[:64]  # BLAKE2b key max 64 bytes
            hash_obj = hashlib.blake2b(data.encode(), key=key_bytes)
        else:
            hash_obj = hashlib.blake2b(data.encode())
        return hash_obj.hexdigest()

    def hmac_sha256(self, key: str, message: str) -> str:
        """Compute HMAC-SHA256"""
        hmac_obj = hmac.new(key.encode(), message.encode(), hashlib.sha256)
        return hmac_obj.hexdigest()

class SecureRandom:
    """Cryptographically secure random number generation"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_hex(length // 2)

    @staticmethod
    def generate_password(length: int = 16, include_special: bool = True) -> str:
        """Generate secure random password"""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if include_special:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"

        return ''.join(secrets.choice(chars) for _ in range(length))

    @staticmethod
    def generate_salt(length: int = 16) -> str:
        """Generate random salt"""
        return base64.b64encode(os.urandom(length)).decode()

class CertificateAuthority:
    """Simple Certificate Authority simulation"""

    def __init__(self):
        self.certificates = {}
        self.revoked = set()

    def issue_certificate(self, public_key_pem: str, subject: str,
                         validity_days: int = 365) -> Dict[str, Any]:
        """Issue a certificate"""
        cert_id = self.generate_cert_id()

        certificate = {
            'id': cert_id,
            'subject': subject,
            'public_key': public_key_pem,
            'issued_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=validity_days)).isoformat(),
            'status': 'active'
        }

        self.certificates[cert_id] = certificate
        return certificate

    def revoke_certificate(self, cert_id: str) -> bool:
        """Revoke a certificate"""
        if cert_id in self.certificates:
            self.certificates[cert_id]['status'] = 'revoked'
            self.certificates[cert_id]['revoked_at'] = datetime.now().isoformat()
            self.revoked.add(cert_id)
            return True
        return False

    def verify_certificate(self, cert_id: str) -> Dict[str, Any]:
        """Verify certificate status"""
        if cert_id not in self.certificates:
            return {'valid': False, 'reason': 'Certificate not found'}

        cert = self.certificates[cert_id]

        if cert['status'] == 'revoked':
            return {'valid': False, 'reason': 'Certificate revoked'}

        expires_at = datetime.fromisoformat(cert['expires_at'])
        if datetime.now() > expires_at:
            return {'valid': False, 'reason': 'Certificate expired'}

        return {'valid': True, 'certificate': cert}

    def generate_cert_id(self) -> str:
        """Generate unique certificate ID"""
        return f"CERT-{secrets.token_hex(8).upper()}"

class SecureMessaging:
    """Secure messaging utilities"""

    def __init__(self):
        self.crypto = AdvancedCrypto()

    def generate_key_exchange(self) -> Dict[str, str]:
        """Generate key exchange parameters (simplified Diffie-Hellman)"""
        # Generate ephemeral key pair
        private_key_pem, public_key_pem = self.crypto.generate_rsa_keypair()

        return {
            'private_key': private_key_pem,
            'public_key': public_key_pem,
            'session_id': secrets.token_hex(16)
        }

    def encrypt_message(self, sender_private_key: str, recipient_public_key: str,
                       message: str) -> Dict[str, str]:
        """Encrypt a message for secure transmission"""
        # Generate session key
        session_key = self.crypto.generate_aes_key()

        # Encrypt message with session key
        encrypted_message, iv = self.crypto.aes_encrypt(session_key, message)

        # Encrypt session key with recipient's public key
        encrypted_session_key = self.crypto.rsa_encrypt(recipient_public_key, session_key)

        # Sign the encrypted message
        signature = self.crypto.ecdsa_sign(sender_private_key, encrypted_message)

        return {
            'encrypted_message': encrypted_message,
            'encrypted_session_key': encrypted_session_key,
            'iv': iv,
            'signature': signature,
            'timestamp': datetime.now().isoformat()
        }

    def decrypt_message(self, recipient_private_key: str, sender_public_key: str,
                       encrypted_data: Dict[str, str]) -> Optional[str]:
        """Decrypt a received message"""
        try:
            # Decrypt session key
            session_key = self.crypto.rsa_decrypt(
                recipient_private_key,
                encrypted_data['encrypted_session_key']
            )

            # Verify signature
            is_valid = self.crypto.ecdsa_verify(
                sender_public_key,
                encrypted_data['encrypted_message'],
                encrypted_data['signature']
            )

            if not is_valid:
                return None

            # Decrypt message
            message = self.crypto.aes_decrypt(
                session_key,
                encrypted_data['encrypted_message'],
                encrypted_data['iv']
            )

            return message

        except Exception as e:
            print(f"Decryption failed: {e}")
            return None

class BlockchainCrypto:
    """Blockchain-related cryptographic functions"""

    def __init__(self):
        self.hash_func = HashFunctions()

    def create_block_hash(self, index: int, previous_hash: str, timestamp: str,
                         data: str, nonce: int) -> str:
        """Create block hash for blockchain"""
        block_string = f"{index}{previous_hash}{timestamp}{data}{nonce}"
        return self.hash_func.sha256_hash(block_string)

    def proof_of_work(self, index: int, previous_hash: str, timestamp: str,
                     data: str, difficulty: int = 4) -> Tuple[int, str]:
        """Simple proof of work"""
        nonce = 0
        target = "0" * difficulty

        while True:
            block_hash = self.create_block_hash(index, previous_hash, timestamp, data, nonce)
            if block_hash.startswith(target):
                return nonce, block_hash
            nonce += 1

    def generate_wallet_address(self, public_key_pem: str) -> str:
        """Generate blockchain wallet address from public key"""
        # Simple address generation (not cryptographically secure for real blockchain)
        public_key_hash = self.hash_func.sha256_hash(public_key_pem)
        address = base64.b32encode(bytes.fromhex(public_key_hash[:40])).decode()[:34]
        return f"BC1{address}"

    def create_transaction(self, sender_address: str, recipient_address: str,
                          amount: float, private_key_pem: str) -> Dict[str, Any]:
        """Create a signed transaction"""
        transaction = {
            'sender': sender_address,
            'recipient': recipient_address,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'tx_id': secrets.token_hex(16)
        }

        # Create transaction string for signing
        tx_string = f"{transaction['sender']}{transaction['recipient']}{transaction['amount']}{transaction['timestamp']}"

        # Sign transaction
        crypto = AdvancedCrypto()
        signature = crypto.ecdsa_sign(private_key_pem, tx_string)

        transaction['signature'] = signature
        return transaction

    def verify_transaction(self, transaction: Dict[str, Any], public_key_pem: str) -> bool:
        """Verify transaction signature"""
        tx_string = f"{transaction['sender']}{transaction['recipient']}{transaction['amount']}{transaction['timestamp']}"

        crypto = AdvancedCrypto()
        return crypto.ecdsa_verify(public_key_pem, tx_string, transaction['signature'])

class QuantumResistantCrypto:
    """Quantum-resistant cryptographic functions (simplified)"""

    def __init__(self):
        self.hash_func = HashFunctions()

    def create_merkle_tree(self, data_list: list) -> Tuple[str, Dict[str, Any]]:
        """Create Merkle tree from data list"""
        if not data_list:
            return self.hash_func.sha256_hash(""), {}

        # Hash all data
        hashed_data = [self.hash_func.sha256_hash(str(data)) for data in data_list]

        tree = {'leaves': hashed_data, 'levels': [hashed_data]}

        # Build tree levels
        current_level = hashed_data
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]  # Duplicate last element
                next_level.append(self.hash_func.sha256_hash(combined))
            tree['levels'].append(next_level)
            current_level = next_level

        tree['root'] = current_level[0]
        return tree['root'], tree

    def generate_lattice_keypair(self, dimension: int = 256) -> Tuple[list, list]:
        """Simplified lattice-based key generation (educational only)"""
        # This is a highly simplified version for educational purposes
        # Real lattice crypto is much more complex

        # Generate random matrices (simplified)
        private_key = [[secrets.randbelow(100) for _ in range(dimension)] for _ in range(dimension)]
        public_key = [[secrets.randbelow(100) for _ in range(dimension)] for _ in range(dimension)]

        # Add some relationship (simplified key generation)
        for i in range(dimension):
            for j in range(dimension):
                public_key[i][j] = (private_key[i][j] * 2 + secrets.randbelow(10)) % 100

        return private_key, public_key

    def lattice_encrypt(self, public_key: list, message: str) -> list:
        """Simplified lattice encryption"""
        # Convert message to numbers
        message_nums = [ord(c) for c in message[:len(public_key)]]

        # Pad if necessary
        while len(message_nums) < len(public_key):
            message_nums.append(0)

        # Simple matrix multiplication (educational)
        ciphertext = []
        for i in range(len(public_key)):
            result = 0
            for j in range(len(message_nums)):
                result += public_key[i][j] * message_nums[j]
            ciphertext.append(result % 100)

        return ciphertext

    def lattice_decrypt(self, private_key: list, ciphertext: list) -> str:
        """Simplified lattice decryption"""
        # Simple matrix multiplication (educational)
        plaintext_nums = []
        for i in range(len(private_key)):
            result = 0
            for j in range(len(ciphertext)):
                result += private_key[j][i] * ciphertext[j]
            plaintext_nums.append(result % 100)

        # Convert back to characters
        plaintext = ''.join(chr(num) if 32 <= num <= 126 else '?' for num in plaintext_nums)
        return plaintext.strip('?')

class CryptoBenchmark:
    """Benchmarking utilities for cryptographic operations"""

    def __init__(self):
        self.crypto = AdvancedCrypto()
        self.hash_func = HashFunctions()

    def benchmark_rsa(self, key_sizes=[1024, 2048, 3072], iterations=10) -> Dict[str, Any]:
        """Benchmark RSA operations"""
        results = {}

        for key_size in key_sizes:
            print(f"Benchmarking RSA-{key_size}...")

            # Generate key pair
            start_time = time.time()
            private_key, public_key = self.crypto.generate_rsa_keypair(key_size)
            keygen_time = time.time() - start_time

            # Test message
            test_message = "This is a test message for benchmarking cryptographic operations."

            # Benchmark encryption
            encrypt_times = []
            for _ in range(iterations):
                start_time = time.time()
                ciphertext = self.crypto.rsa_encrypt(public_key, test_message)
                encrypt_times.append(time.time() - start_time)

            # Benchmark decryption
            decrypt_times = []
            for _ in range(iterations):
                start_time = time.time()
                plaintext = self.crypto.rsa_decrypt(private_key, ciphertext)
                decrypt_times.append(time.time() - start_time)

            results[key_size] = {
                'key_generation': keygen_time,
                'encryption_avg': sum(encrypt_times) / len(encrypt_times),
                'decryption_avg': sum(decrypt_times) / len(decrypt_times),
                'encryption_min': min(encrypt_times),
                'encryption_max': max(encrypt_times),
                'decryption_min': min(decrypt_times),
                'decryption_max': max(decrypt_times)
            }

        return results

    def benchmark_aes(self, key_sizes=[128, 192, 256], modes=['CBC', 'GCM'], iterations=100) -> Dict[str, Any]:
        """Benchmark AES operations"""
        results = {}

        test_message = "A" * 1000  # 1KB test message

        for key_size in key_sizes:
            for mode in modes:
                print(f"Benchmarking AES-{key_size}-{mode}...")

                # Generate key
                key = self.crypto.generate_aes_key(key_size)

                # Benchmark encryption
                encrypt_times = []
                for _ in range(iterations):
                    start_time = time.time()
                    ciphertext, iv = self.crypto.aes_encrypt(key, test_message, mode)
                    encrypt_times.append(time.time() - start_time)

                # Benchmark decryption
                decrypt_times = []
                for _ in range(iterations):
                    start_time = time.time()
                    plaintext = self.crypto.aes_decrypt(key, ciphertext, iv, mode)
                    decrypt_times.append(time.time() - start_time)

                results[f'AES-{key_size}-{mode}'] = {
                    'encryption_avg': sum(encrypt_times) / len(encrypt_times),
                    'decryption_avg': sum(decrypt_times) / len(decrypt_times),
                    'throughput_mbps': (len(test_message) * iterations) / (1024 * 1024 * sum(encrypt_times + decrypt_times))
                }

        return results

    def benchmark_hash_functions(self, algorithms=['sha256', 'sha3_256', 'blake2b'], iterations=1000) -> Dict[str, Any]:
        """Benchmark hash functions"""
        results = {}

        test_data = "A" * 10000  # 10KB test data

        for algo in algorithms:
            print(f"Benchmarking {algo}...")

            hash_times = []
            for _ in range(iterations):
                start_time = time.time()
                if algo == 'sha256':
                    self.hash_func.sha256_hash(test_data)
                elif algo == 'sha3_256':
                    self.hash_func.sha3_256_hash(test_data)
                elif algo == 'blake2b':
                    self.hash_func.blake2b_hash(test_data)
                hash_times.append(time.time() - start_time)

            results[algo] = {
                'avg_time': sum(hash_times) / len(hash_times),
                'min_time': min(hash_times),
                'max_time': max(hash_times),
                'throughput_mbps': (len(test_data) * iterations) / (1024 * 1024 * sum(hash_times))
            }

        return results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("Starting comprehensive cryptographic benchmark...")

        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'rsa_benchmark': self.benchmark_rsa(),
            'aes_benchmark': self.benchmark_aes(),
            'hash_benchmark': self.benchmark_hash_functions()
        }

        print("Benchmark completed!")
        return benchmark_results
