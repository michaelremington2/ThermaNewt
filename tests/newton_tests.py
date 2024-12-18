import unittest
import math
from rattle_newton.sim_snake_tb import ThermalSimulator

class TestThermalSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = ThermalSimulator(flip_logic='preferred', t_pref_min=25, t_pref_max=35, t_pref_opt=30, seed=42)

    def test_rate_of_heat_transfer_k(self):
        k = 0.5
        t_body = 35
        t_env = 25
        self.assertAlmostEqual(self.simulator.rate_of_heat_transfer_k(k, t_body, t_env), 5.0)

    def test_cooling_eq_k(self):
        t_body = 35
        t_env = 25
        delta_t = 1
        k = 0.1
        expected = t_env + (t_body - t_env) * math.exp(-k * delta_t)
        result = self.simulator.cooling_eq_k(k, t_body, t_env, delta_t)
        self.assertAlmostEqual(result, expected)

    def test_snake_surface_area(self):
        radius = 2
        length = 10
        expected_area = 2 * math.pi * radius * length + 2 * math.pi * radius**2
        self.assertAlmostEqual(self.simulator.snake_surface_area(radius, length), expected_area)

    def test_do_i_flip_preferred(self):
        t_body = 28
        burrow_temp = 20
        open_temp = 35
        result = self.simulator.do_i_flip(t_body, burrow_temp, open_temp)
        self.assertIn(result, ['In', 'Out'])

    def test_tb_simulator_2_state_model_wrapper(self):
        burrow_temp_vector = [20, 22, 24, 26, 28]
        open_temp_vector = [35, 34, 33, 32, 31]
        t_initial = 30
        k = 0.1
        delta_t = 1
        burrow_usage, simulated_t_body = self.simulator.tb_simulator_2_state_model_wrapper(
            k, t_initial, delta_t, burrow_temp_vector, open_temp_vector, return_tbody_sim=True)
        self.assertEqual(len(burrow_usage), len(burrow_temp_vector))
        self.assertEqual(len(simulated_t_body), len(open_temp_vector))


if __name__ == '__main__':
    unittest.main()
