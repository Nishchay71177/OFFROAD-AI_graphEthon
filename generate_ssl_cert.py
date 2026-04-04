#!/usr/bin/env python3
"""
Generate self-signed SSL certificates for localhost and IP addresses
Run this before starting the app with HTTPS
"""

import subprocess
import sys
import os
from pathlib import Path

def generate_certificate():
    """Generate self-signed certificate using OpenSSL"""
    
    cert_dir = Path(__file__).parent / 'certs'
    cert_dir.mkdir(exist_ok=True)
    
    cert_file = cert_dir / 'cert.pem'
    key_file = cert_dir / 'key.pem'
    
    # Check if certificates already exist
    if cert_file.exists() and key_file.exists():
        print(f"✅ Certificates already exist at {cert_dir}")
        return str(cert_file), str(key_file)
    
    print("🔐 Generating self-signed SSL certificate for HTTPS...")
    print(f"📁 Certificate directory: {cert_dir}")
    
    try:
        # OpenSSL command for self-signed certificate valid for 365 days
        # Includes localhost and common IP addresses
        cmd = [
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
            '-keyout', str(key_file),
            '-out', str(cert_file),
            '-days', '365',
            '-nodes',
            '-subj', '/CN=localhost'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"✅ Certificate generated successfully!")
            print(f"   Cert: {cert_file}")
            print(f"   Key:  {key_file}")
            return str(cert_file), str(key_file)
        else:
            print(f"❌ OpenSSL error: {result.stderr}")
            return None, None
            
    except FileNotFoundError:
        print("❌ OpenSSL not found. Installing via Python cryptography...")
        generate_with_cryptography(cert_file, key_file)
        if cert_file.exists() and key_file.exists():
            return str(cert_file), str(key_file)
        return None, None

def generate_with_cryptography(cert_file, key_file):
    """Fallback: Generate certificate using Python cryptography library"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        import ipaddress
        
        print("📦 Using cryptography library to generate certificate...")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.DNSName(u"127.0.0.1"),
                x509.DNSName(u"*.local"),
                x509.IPAddress(ipaddress.IPv4Address(u"127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Write certificate
        with open(cert_file, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        # Write private key
        with open(key_file, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        print(f"✅ Certificate generated with cryptography!")
        print(f"   Cert: {cert_file}")
        print(f"   Key:  {key_file}")
        
    except ImportError:
        print("❌ cryptography library not found. Install it:")
        print("   pip install cryptography")

if __name__ == '__main__':
    cert, key = generate_certificate()
    
    if cert and key:
        print("\n✅ Ready for HTTPS!")
        print(f"\nUsage in app.py:")
        print(f"  app.run(ssl_context=('{cert}', '{key}'))")
        sys.exit(0)
    else:
        print("\n❌ Failed to generate certificate")
        sys.exit(1)
