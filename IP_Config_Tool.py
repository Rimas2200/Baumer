#!/usr/bin/env python3

"""
Функция "Baumer IP Config Tool", используется для обнаружения
и настройки IP-адресов камер Baumer в сети. В её основе лежат протоколы и механизмы
для обнаружения устройств в сети, такие как DHCP, ARP, и протоколы для
конфигурации устройств. Код позволяет обнаружить камеры во всех интерфейсах.
"""

from pypylon import pylon
import psutil
import socket

# Функция для получения IP-адресов всех интерфейсов
def get_all_interface_ips():
    addresses = psutil.net_if_addrs()
    interface_ips = {}
    for interface_name, addr_list in addresses.items():
        for addr in addr_list:
            if addr.family == socket.AF_INET:
                interface_ips[interface_name] = addr.address
    return interface_ips

# Получить IP-адреса всех интерфейсов
interface_ips = get_all_interface_ips()
print("Сетевые интерфейсы и их IP-адреса:")
for interface, ip in interface_ips.items():
    print(f" - {interface}: {ip}")

# Инициализация транспортного слоя
tl_factory = pylon.TlFactory.GetInstance()

# Создание экземпляра транспортного слоя для GigE
gige_tl = pylon.TlFactory.GetInstance().CreateTl('BaslerGigE')

# Поиск устройств через транспортный слой
devices = gige_tl.EnumerateDevices()
print(f"Найдено {len(devices)} камера(ы):")
for i, device in enumerate(devices):
    device_ip = device.GetIpAddress()
    matching_interface = None
    for interface, ip in interface_ips.items():
        if ip.startswith('192.168.1.') and device_ip.startswith('192.168.1.'):
            matching_interface = interface
            break
    if matching_interface:
        print(f"Камера {i} найдена на интерфейсе {matching_interface}:")
    else:
        print(f"Камера {i}:")
    print(f"  - Модель: {device.GetModelName()}")
    print(f"  - Производитель: {device.GetVendorName()}")
    print(f"  - Серийный номер: {device.GetSerialNumber()}")
    print(f"  - IP-адрес: {device.GetIpAddress()}")

    camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
    camera.Open()
    print(f"Подключено к камере: {camera.GetDeviceInfo().GetModelName()}")

    camera.Close()
