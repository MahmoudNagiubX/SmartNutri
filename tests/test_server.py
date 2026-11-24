import unittest

from smartnutri_backend.server import resolve_bind_host, resolve_port


class ServerBindingTests(unittest.TestCase):
    def test_bind_defaults_to_loopback(self):
        self.assertEqual(resolve_bind_host(None), "127.0.0.1")
        self.assertEqual(resolve_port(None), 8000)

    def test_private_ipv4_and_loopback_addresses_are_accepted(self):
        for address in ("127.0.0.1", "10.0.0.12", "172.16.2.5", "192.168.1.20"):
            with self.subTest(address=address):
                self.assertEqual(resolve_bind_host(address), address)

    def test_public_wildcard_hostname_and_non_ipv4_addresses_are_rejected(self):
        for address in ("0.0.0.0", "8.8.8.8", "localhost", "::1", "224.0.0.1"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                resolve_bind_host(address)

    def test_port_must_be_an_integer_in_the_tcp_range(self):
        for value in ("0", "65536", "eight", "80.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_port(value)
        self.assertEqual(resolve_port("8123"), 8123)


if __name__ == "__main__":
    unittest.main()
