import unittest
import torch
import unitair.states as states
import random
import unitair


class VectorTensorConversions(unittest.TestCase):

    def test_vector_tensor_vector(self):
        n_min = 5
        n_max = 13

        n_qubits = random.randint(n_min, n_max)
        real_vec = torch.rand(2 ** n_qubits)
        vec_batched = torch.rand(7, 1, 4, 2 ** n_qubits)

        tens = states.to_tensor_layout(real_vec)
        tens_batched = states.to_tensor_layout(vec_batched)

        vec_back = states.to_vector_layout(
            state_tensor=tens,
            num_qubits=n_qubits
        )
        vec_back_batched = states.to_vector_layout(
            state_tensor=tens_batched,
            num_qubits=n_qubits
        )

        self.assertTrue(
            (vec_back == real_vec).all()
        )

        self.assertTrue(
            (vec_back_batched == vec_batched).all()
        )


    def test_tensor_vector_tensor(self):

        n_min = 5
        n_max = 13
        n = random.randint(n_min, n_max)
        tens = torch.rand(n * (2,))

        tens_batch = torch.rand((1, 2, 3,) + n * (2,))
        real_vec = states.to_vector_layout(tens, n)
        vec_batch = states.to_vector_layout(tens_batch, n)
        real_tens_back = states.to_tensor_layout(real_vec)
        tens_back_batch = states.to_tensor_layout(vec_batch)

        self.assertTrue((tens == real_tens_back).all())

        self.assertTrue((tens_batch == tens_back_batch).all())


class TestQubitCounting(unittest.TestCase):

    def test_count_qubits(self):
        state = torch.rand(2 ** 4)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 4)

        state = torch.rand(7, 4, 1, 2 ** 5)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 5)

        state = torch.rand(2, 2, 2, 2, 2)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 1)

        state_tensor = torch.rand(4, 3, 2, 2, 2, 2, 2)
        state = unitair.states.to_vector_layout(state_tensor, num_qubits=3)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 3)

    def test_qubit_indices(self):
        n = 4
        state = torch.rand(5, 2**n)
        ind_selected = states.shapes.get_qubit_indices(
            index=3,
            state_tensor=states.to_tensor_layout(state),
            num_qubits=n
        )
        self.assertEqual(4, ind_selected)

        n = 5
        state = torch.rand(2, 5, 7, 2 ** n)
        ind_selected = states.shapes.get_qubit_indices(
            index=torch.tensor([2, -3, 1, 0]),
            state_tensor=states.to_tensor_layout(state),
            num_qubits=n
        )
        expected = torch.tensor([5, -3, 4, 3])
        self.assertTrue(expected.eq(ind_selected).all())

        n = 4
        state = torch.rand(2, 2, 2 ** n)
        ind_selected = states.shapes.get_qubit_indices(
            index=[-2, -4, 0, 1],
            state_tensor=states.to_tensor_layout(state),
            num_qubits=n
        )
        expected = [-2, -4, 2, 3]
        self.assertEqual(expected, ind_selected)


if __name__ == '__main__':
    unittest.main()
