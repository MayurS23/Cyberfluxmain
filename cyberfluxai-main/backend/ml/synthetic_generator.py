import numpy as np
import random
from datetime import datetime, timezone
from typing import Dict, Any


class SyntheticTrafficGenerator:
    """Generate synthetic network traffic for simulation mode"""
    
    def __init__(self):
        self.attack_types = [
            "DDoS", "Port Scan", "Brute Force", "SQL Injection",
            "XSS", "Botnet", "Infiltration", "DoS", "Normal"
        ]
        self.protocols = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS"]
        self.src_ips = [f"192.168.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(20)]
        self.dst_ips = [f"10.0.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(10)]
    
    def generate_flow(self) -> Dict[str, Any]:
        """Generate a single network flow"""
        is_attack = random.random() > 0.7
        
        if is_attack:
            attack_type = random.choice(self.attack_types[:-1])  # Exclude "Normal"
            severity = random.randint(60, 100)
            packet_count = random.randint(1000, 50000)
            byte_count = random.randint(100000, 10000000)
        else:
            attack_type = "Normal"
            severity = random.randint(0, 40)
            packet_count = random.randint(10, 1000)
            byte_count = random.randint(1000, 100000)
        
        flow = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "src_ip": random.choice(self.src_ips),
            "dst_ip": random.choice(self.dst_ips),
            "src_port": random.randint(1024, 65535),
            "dst_port": random.choice([22, 80, 443, 3389, 8080]),
            "protocol": random.choice(self.protocols),
            "packet_count": packet_count,
            "byte_count": byte_count,
            "duration": round(random.uniform(0.1, 300.0), 2),
            "attack_type": attack_type,
            "severity": severity,
            "is_attack": is_attack,
            "confidence": round(random.uniform(70, 98), 2)
        }
        
        return flow
    
    def generate_batch(self, count: int = 10) -> list:
        """Generate multiple flows"""
        return [self.generate_flow() for _ in range(count)]


# Global generator instance
generator = SyntheticTrafficGenerator()