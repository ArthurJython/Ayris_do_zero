#!/usr/bin/env python3
"""Servidor local somente de leitura para o AYRIS Monitor (Linux)."""
from __future__ import annotations

import json
import re
import shutil
import socket
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HOST = "0.0.0.0"  # permite abrir no celular pela mesma rede Wi-Fi
PORT = 8080
_previous_cpu = None


def cpu_usage_percent():
    """Calcula uso total da CPU entre duas leituras de /proc/stat."""
    global _previous_cpu
    try:
        with open("/proc/stat", "r", encoding="ascii") as f:
            fields = f.readline().split()
        if not fields or fields[0] != "cpu":
            return None
        values = [int(x) for x in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        current = (total, idle)
        previous = _previous_cpu
        _previous_cpu = current
        if previous is None:
            return None
        total_delta = current[0] - previous[0]
        idle_delta = current[1] - previous[1]
        if total_delta <= 0:
            return None
        return round(100.0 * (total_delta - idle_delta) / total_delta, 1)
    except (OSError, ValueError, IndexError):
        return None


def read_cpu_temperature():
    """Lê sensores hwmon/thermal; prioriza Package/Tdie/Tctl da CPU."""
    candidates = []

    # hwmon costuma expor os sensores coretemp do Intel.
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            chip = (hwmon / "name").read_text().strip().lower()
        except OSError:
            chip = ""
        if not any(key in chip for key in ("coretemp", "k10temp", "zenpower", "cpu", "soc")):
            continue
        for input_file in hwmon.glob("temp*_input"):
            stem = input_file.name.removesuffix("_input")
            try:
                milli_c = float(input_file.read_text().strip())
                temp_c = milli_c / 1000.0
                if not (-10 <= temp_c <= 150):
                    continue
                try:
                    label = (hwmon / f"{stem}_label").read_text().strip().lower()
                except OSError:
                    label = ""
                priority = 0 if any(x in label for x in ("package", "tdie", "tctl")) else 1
                candidates.append((priority, temp_c, f"{chip}/{label or stem}"))
            except (OSError, ValueError):
                pass

    # Alguns kernels expõem a temperatura em thermal_zone.
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            zone_type = (zone / "type").read_text().strip().lower()
            raw = float((zone / "temp").read_text().strip())
            temp_c = raw / 1000.0 if abs(raw) > 200 else raw
            if not (-10 <= temp_c <= 150):
                continue
            if any(x in zone_type for x in ("cpu", "x86_pkg", "soc", "package")):
                candidates.append((0, temp_c, f"thermal/{zone_type}"))
        except (OSError, ValueError):
            pass

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], -item[1]))
    _, temperature, source = candidates[0]
    return round(temperature, 1), source


def read_memory():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            match = re.match(r"(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
        total = values["MemTotal"]
        available = values["MemAvailable"]
        used = total - available
        return {
            "ram_total_gb": round(total / (1024 ** 3), 2),
            "ram_used_gb": round(used / (1024 ** 3), 2),
            "ram_percent": round(100 * used / total, 1) if total else 0,
        }
    except (OSError, KeyError, ValueError):
        return {"ram_total_gb": None, "ram_used_gb": None, "ram_percent": None}


def get_hardware():
    temp, temp_source = read_cpu_temperature()
    try:
        disk = shutil.disk_usage("/")
        disk_total = disk.total
        disk_used = disk.used
        disk_info = {
            "disk_total_gb": round(disk_total / (1024 ** 3), 1),
            "disk_used_gb": round(disk_used / (1024 ** 3), 1),
            "disk_percent": round(100 * disk_used / disk_total, 1) if disk_total else 0,
        }
    except OSError:
        disk_info = {"disk_total_gb": None, "disk_used_gb": None, "disk_percent": None}

    return {
        "timestamp": time.time(),
        "cpu_percent": cpu_usage_percent(),
        "cpu_temp_c": temp,
        "cpu_temp_source": temp_source,
        **read_memory(),
        **disk_info,
        "gpu_percent": None,  # esta máquina está treinando na CPU
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/hardware":
            payload = json.dumps(get_hardware()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, fmt, *args):
        # Evita poluir o terminal com uma linha a cada polling.
        pass


def main():
    # Primeira amostra de CPU; a segunda requisição já calcula a porcentagem.
    cpu_usage_percent()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"AYRIS Monitor: http://127.0.0.1:{PORT}")
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
        ips = sorted({item[4][0] for item in infos if not item[4][0].startswith("127.")})
        for ip in ips:
            print(f"Celular na mesma rede: http://{ip}:{PORT}")
    except OSError:
        pass
    print("Somente leitura: não acessa nem altera o processo de treinamento.")
    print("Ctrl+C encerra apenas o monitor.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\\nMonitor encerrado.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
