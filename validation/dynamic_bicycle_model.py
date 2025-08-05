import jax.numpy as jnp




class FynamicBicycleModel:

    def __init__(self, params):
        self.params  = params

    


    def Dynamics(self, state, inputs, params=None):
        input_neural_states = state[3:7]  # (4,)
        nn_inputs = jnp.concatenate((input_neural_states, inputs))  # (6,)
        assert nn_inputs.shape == (6,), f"nn_inputs shape is {nn_inputs.shape}, expected (6,)"
        if params is None:
            params = self.params
        neural_output = self.neural_net.apply(params, nn_inputs)
        return neural_output
    

    
    
    def rk4_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None):
        """
        Perform a single RK4 integration step for the hybrid ODE.
        Args:
            params: neural network parameters
            state: current state vector
            inputs_t: inputs at time t
            inputs_t_plus_dt: inputs at time t + dt
            dt: time step (float)
        Returns:
            next_state: state vector after time dt
        """
        inputs_mid = (inputs_t + inputs_t_plus_dt) / 2.0

        k1 = self.hybrid_dynamics(state, inputs_t, params)
        k2 = self.hybrid_dynamics(state + dt/2 * k1, inputs_mid, params)
        k3 = self.hybrid_dynamics(state + dt/2 * k2, inputs_mid, params)
        k4 = self.hybrid_dynamics(state + dt * k3, inputs_t_plus_dt, params)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Wrap yaw (index 2) to [-pi, pi] after integration
        next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return next_state
    

    

    def euler_step(self, state, inputs_t, inputs_t_plus_dt, dt, params=None):
        """
        Perform a single Euler integration step for the hybrid ODE.
        Args:
            params: neural network parameters
            state: current state vector
            inputs_t: inputs at time t
            inputs_t_plus_dt: (unused, for API compatibility)
            dt: time step (float)
        Returns:
            next_state: state vector after time dt
        """
        k1 = self.hybrid_dynamics(state, inputs_t, params)
   
        next_state = state + dt * k1
        # Wrap yaw (index 2) to [-pi, pi] after integration
        next_state = next_state.at[2].set(((next_state[2] + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        return next_state

    def predict_trajectory(self, params, initial_state, inputs_sequence, dt):
        """
        Roll out the trajectory for num_steps-1 using the given initial state and input sequence,
        using jax.lax.scan for efficiency. Only predicts up to the last available ground-truth state.
        Args:
            initial_state: shape (state_dim,)
            inputs_sequence: shape (num_steps, input_dim)
            dt: float, time step
        Returns:
            states: shape (num_steps, state_dim)
        """
        num_steps = inputs_sequence.shape[0]

        def scan_step(state, t):
            current_input = inputs_sequence[t]
            next_input = inputs_sequence[t + 1]  
            next_state = self.rk4_step(state, current_input, next_input, dt, params)
            return next_state, next_state

        
        indices = jnp.arange(num_steps - 1)
        _, states = jax.lax.scan(scan_step, initial_state, indices)
        
        trajectory = jnp.vstack([initial_state, states])
        return trajectory
    
    


    def predict_batch_trajectories(self, params, initial_states, inputs_batch, dt):
        batch_predict_fn = jax.vmap(lambda s, i: self.predict_trajectory(params, s, i, dt), in_axes=(0, 0))
        return batch_predict_fn(initial_states, inputs_batch)
