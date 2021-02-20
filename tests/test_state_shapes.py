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
        complex_vec = torch.rand(2, 2 ** n_qubits)
        real_vec_batched = torch.rand(7, 1, 4, 2 ** n_qubits)
        complex_vec_batched = torch.rand(7, 1, 4, 2, 2 ** n_qubits)

        real_tens = states.to_tensor_layout(real_vec)
        complex_tens = states.to_tensor_layout(complex_vec)
        real_tens_batched = states.to_tensor_layout(real_vec_batched)
        complex_tens_batched = states.to_tensor_layout(complex_vec_batched)

        real_vec_back = states.to_vector_layout(
            state_tensor=real_tens,
            num_qubits=n_qubits
        )
        complex_vec_back = states.to_vector_layout(
            state_tensor=complex_tens,
            num_qubits=n_qubits
        )
        real_vec_back_batched = states.to_vector_layout(
            state_tensor=real_tens_batched,
            num_qubits=n_qubits
        )
        complex_vec_back_batched = states.to_vector_layout(
            state_tensor=complex_tens_batched,
            num_qubits=n_qubits
        )
        self.assertTrue(
            (real_vec_back == real_vec).all()
        )
        self.assertTrue(
            (complex_vec_back == complex_vec).all()
        )
        self.assertTrue(
            (real_vec_back_batched == real_vec_batched).all()
        )
        self.assertTrue(
            (complex_vec_back_batched == complex_vec_batched).all()
        )

    def test_tensor_vector_tensor(self):

        n_min = 5
        n_max = 13
        n = random.randint(n_min, n_max)
        real_tens = torch.rand(n * (2,))
        complex_tens = torch.rand((2,) + n * (2,))
        real_tens_batch = torch.rand((1, 2, 3,) + n * (2,))
        complex_tens_batch = torch.rand((1, 2, 3,) + (2,) + n * (2,))

        real_vec = states.to_vector_layout(real_tens, n)
        complex_vec = states.to_vector_layout(complex_tens, n)
        real_vec_batch = states.to_vector_layout(real_tens_batch, n)
        complex_vec_batch = states.to_vector_layout(complex_tens_batch, n)

        real_tens_back = states.to_tensor_layout(real_vec)
        complex_tens_back = states.to_tensor_layout(complex_vec)
        real_tens_back_batch = states.to_tensor_layout(real_vec_batch)
        complex_tens_back_batch = states.to_tensor_layout(complex_vec_batch)

        self.assertTrue((real_tens == real_tens_back).all())
        self.assertTrue((complex_tens == complex_tens_back).all())

        self.assertTrue((real_tens_batch == real_tens_back_batch).all())
        self.assertTrue((complex_tens_batch == complex_tens_back_batch).all())


class TestQubitCounting(unittest.TestCase):

    def test_count_qubits(self):
        state = torch.rand(2, 2 ** 4)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 4)

        state = torch.rand(2 ** 5)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 5)

        state = torch.rand(2, 2, 2, 2, 2)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 1)

        state_tensor = torch.rand(4, 3, 2, 2, 2, 2, 2)
        state = unitair.states.to_vector_layout(state_tensor, 3)
        n = unitair.states.count_qubits(state)
        self.assertEqual(n, 3)

    def qubit_indices(self):
        n = 4
        state = torch.rand(2, 2**n)
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
        self.assertTrue(expected.eq(ind_selected))

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
