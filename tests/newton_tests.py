import unittest
from rattle_newton.sim_snake_tb import ThermalSimulator


class TestThermalSimulator(unittest.TestCase):

    def setUp(self):
        # Initialize necessary parameters for testing ThermalSimulator
        self.simulator = ThermalSimulator(flip_logic='random', 
                                          t_pref_min=10, 
                                          t_pref_max=30, 
                                          t_pref_opt=20, 
                                          seed=42)

    def test_random_flips(self):
        # Test random_flips method
        flips = [self.simulator.random_flips() for _ in range(100)]
        unique_flips = set(flips)
        self.assertIn('In', unique_flips)
        self.assertIn('Out', unique_flips)

    def test_do_i_flip(self):
        # Test do_i_flip method
        bu = self.simulator.do_i_flip(25, 15, 25)  # Test when t_body = 25, burrow_temp = 15, open_temp = 25
        self.assertIn(bu, ['In', 'Out'])

    def test_tb_simulator_2_state_model_wrapper(self):
        # Test tb_simulator_2_state_model_wrapper method
        k = 0.05
        t_initial = 30
        burrow_temp_vector = [20, 25, 18]
        open_temp_vector = [25, 20, 25]
        
        result = self.simulator.tb_simulator_2_state_model_wrapper(k, t_initial, 
                                                                    burrow_temp_vector, open_temp_vector)
        self.assertEqual(len(result), 3)  # Number of time steps should match input vectors

if __name__ == '__main__':
    unittest.main()
