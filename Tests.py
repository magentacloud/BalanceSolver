import json
import unittest
import numpy as np

from cvxopt.base import matrix

from Main import Solver


class MyTestCase(unittest.TestCase):
    f = open('graph1.json', 'r')
    string = f.read()
    solver = Solver(string)

    incidence_matrix = np.array([[1.0, -1.0, -1.0, 0, 0, 0, 0],
                             [0, -0, 1.0, -1.0, -1.0, 0, 0],
                             [0, 0, 0, 0, 1.0, -1.0, -1.0]])
    tols_vector = np.array([[0.00200, 0.00121, 0.00683, 0.00040, 0.00102, 0.00081, 0.00020]]).transpose()
    flows_vector = np.array([[10.005, 3.003, 6.831, 1.985, 5.093, 4.057, 0.991]])
    result_vector = np.array([[10.035898, 2.99186538, 7.044033, 1.983064, 5.060969, 4.0692388, 0.99173026]])
    result_vector_with_limitations = np.array([[8.2454346, 0.82454253, 7.420892, 2.08640457,
                                                5.3344875, 4.3273560, 1.0071314]])
    flows_vector_1 = np.array([[7.33, 3.003, 6.831, 6.666, 5.093, 1.337, 0.991]])
    A_for_negative_test = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    b_for_negative_test = np.array([3.0, 5.0])
    A_for_negative_test_1 = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]])
    b_for_negative_test_1 = np.array([4.0, 4.0, 1.0, 1.0])
    A_for_negative_test_2 = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    b_for_negative_test_2 = np.array([4.0, -4.0, 1.0, -1.0, -30.0, -1.0])
    b_for_negative_test_22 = np.array([4.0, 4.0, 1.0, 1.0, 2.0, 2.0])

    def test_create_matrix_from_graph(self):
        testing_matrix = self.solver.create_incidence_matrix_from_graph(self.solver.data['graph'])
        np.testing.assert_allclose(self.incidence_matrix, testing_matrix)

    def test_create_tols_vector(self):
        testing_vector = self.solver.create_tols_vector(self.solver.data['graph'])
        np.testing.assert_allclose(self.tols_vector, testing_vector, atol=1e-5)

    def test_create_flows_vector(self):
        testing_vector = self.solver.create_flows_vector(self.solver.data['graph'])
        np.testing.assert_allclose(self.flows_vector.transpose(), testing_vector, atol=1e-5)

    def test_solving(self):
        t = self.solver.calculate(self.tols_vector, self.flows_vector.transpose(), self.incidence_matrix, np.zeros(self.solver.data['graph']['nodeCount']))
        np.testing.assert_allclose(self.result_vector, np.array(t).transpose(), atol=1e-3)

    def test_solving_with_limitations(self):
        limitation_vector = matrix(self.solver.data['limitations']['vector'], (1, self.solver.data['graph']['edgesCount']), 'd')
        limitation_rhs = matrix(self.solver.data['limitations']['rhs'], (1, 1), 'd')
        t = self.solver.calculate(self.tols_vector, self.flows_vector.transpose(), self.incidence_matrix,
                                  np.zeros(self.solver.data['graph']['nodeCount']), limitation_vector, limitation_rhs)
        np.testing.assert_allclose(self.result_vector_with_limitations, np.array(t).transpose(), atol=1e-2)

    def test_negative(self):
        t = self.solver.calculate(self.tols_vector, self.flows_vector.transpose(), self.incidence_matrix,
                                  np.zeros(self.solver.data['graph']['nodeCount']),
                                  matrix(self.A_for_negative_test_2.transpose()),
                                  matrix(self.b_for_negative_test_2.transpose()))
        self.assertEqual(t, None)
if __name__ == '__main__':
    unittest.main()
