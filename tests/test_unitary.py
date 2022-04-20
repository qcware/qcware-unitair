import unittest
import torch
import unitair.simulation as sim
import unitair.gates as gates
import unitair.initializations as initializations
import unitair.states as states


class UnitaryOperatorCases(unittest.TestCase):
    def test_phases(self):

        def basic_phase(angles_: torch.Tensor, state_: torch.Tensor):
            """Multiply the jth dimension of state by e^(-i angles_j)."""
            cos = angles_.cos()
            sin = angles_.sin()
            return torch.complex(
                cos * state.real + sin * state.imag,
                -sin * state.real + cos * state.imag
            )

        state = initializations.rand_state(5)
        angles = torch.rand(2 ** 5)

        rotated = sim.apply_phase(angles, state)
        rotated_basic = basic_phase(angles, state)

        self.assertTrue((torch.isclose(rotated, rotated_basic)).all())

        state_batch = torch.rand(4, 2, 2 ** 3)
        angles = torch.rand(4, 2, 2 ** 3)
        rotated = sim.apply_phase(angles, state_batch)

        single_entry_rotated = sim.apply_phase(angles[2, 0], state_batch[2, 0])

        self.assertEqual(rotated.size(), torch.Size([4, 2, 2 ** 3]))
        self.assertTrue((rotated[2, 0] == single_entry_rotated).all())

    def test_apply_to_apply_all(self):
        for n in (1, 7):

            rot = gates.exp_x(.3)

            init_state = initializations.rand_state(n)
            final_1 = sim.apply_all_qubits(rot, init_state)

            final_2 = sim.apply_to_qubits(
                    operators=[rot for _ in range(n)],
                    qubits=range(n),
                    state=init_state
                )

            self.assertTrue((final_1 == final_2).all())

    def test_apply_last(self):
        for n in range(1, 6):

            op = torch.rand(2, 2, dtype=torch.complex64)

            state = initializations.rand_state(n)
            final_1 = sim.act_last_qubit(op, state)
            final_2 = sim.apply_to_qubits([op], [n - 1], state)

            self.assertTrue((final_1 == final_2).all())

        for n in range(1, 5):
            op = torch.rand(2, 2, dtype=torch.complex64)

            state = initializations.rand_state(n)
            final_1 = sim.act_last_qubit(op, state)
            final_2 = sim.apply_to_qubits([op], [n - 1], state)

            self.assertTrue((final_1 == final_2).all())

        op = torch.rand(2, 2, dtype=torch.complex64)
        state_batch = torch.rand(4, 3, 2, 2, 2, 2, 2, dtype=torch.complex64)
        final_batch = sim.act_last_qubit(op, state_batch)
        same = torch.allclose(
            final_batch[2, 2],
            sim.act_last_qubit(op, state_batch[2, 2])
        )
        self.assertTrue(
            same,
            msg='Problem with batching behavior for act_last_qubit.'
        )
        same = torch.allclose(
            final_batch[0, 1],
            sim.act_last_qubit(op, state_batch[0, 1])
        )
        self.assertTrue(
            same,
            msg='Problem with batching behavior for act_last_qubit.'
        )

    def test_swap(self):
        # First check some invariant examples.
        state = torch.tensor([.3, 0., 0., -0.13])
        swapped = sim.swap(state, qubit_pair=(0, 1))
        self.assertTrue(swapped.allclose(state))
        # complex invariant example
        state = torch.tensor([2.24 + .3j, 1. + .2j, 1. + .2j, 73. - .13j])
        swapped = sim.swap(state, qubit_pair=(0, 1))
        self.assertTrue(swapped.allclose(state))

        # random example with five qubits:
        state = initializations.rand_state(5)
        swapped = sim.swap(state, (0, 4))
        state_tensor = states.to_tensor_layout(state)
        manual_swap = state_tensor.transpose(0, 4)
        manual_swap = states.to_vector_layout(manual_swap, num_qubits=5)
        self.assertTrue(swapped.allclose(manual_swap))

        # Now a complicated example with batch dimensions for four qubits:
        state = torch.rand(3, 1, 3, 2 ** 4, dtype=torch.complex64)
        swapped = sim.swap(state, (1, 2))

        state_tensor = states.to_tensor_layout(state)
        manual_swap = state_tensor.transpose(4, 5)
        manual_swap = states.to_vector_layout(manual_swap, num_qubits=4)
        self.assertTrue(swapped.allclose(manual_swap))

        # Check batching behavior
        state = torch.rand(2, 3, 2 ** 4, dtype=torch.complex64)
        swapped = sim.swap(state, qubit_pair=(3, 1))
        swapped_entry = sim.swap(state[0, 1], qubit_pair=(3, 1))
        self.assertTrue(swapped[0, 1].allclose(swapped_entry))
        swapped_entry = sim.swap(state[1, 1], qubit_pair=(3, 1))
        self.assertTrue(swapped[1, 1].allclose(swapped_entry))



if __name__ == '__main__':
    unittest.main()
