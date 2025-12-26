import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==================== HILL FUNCTIONS ====================
def get_clock(t, amp=100, per=24, phase=0):
    """Clock generator matching the project style"""
    return amp * (np.sin(2 * np.pi * t / per + phase) + 1) / 2

def activate_1(A, Kd, n):
    """1 input activation: YES gate"""
    return pow(A/Kd, n) / (1 + pow(A/Kd, n))

def repress_1(R, Kd, n):
    """1 input inhibition: NOT gate"""
    return 1 / (1 + pow(R/Kd, n))

# ==================== FLIP-FLOP MODEL ====================
def ff_ode_model(Y, T, params):
    """Master-slave D flip-flop ODE model"""
    a, not_a, q, not_q, d, clk = Y
    alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n = params

    da_dt = (alpha2 * (pow(d/Kd, n) / (1 + pow(d/Kd, n) + pow(clk/Kd, n) + 
             pow(d/Kd, n)*pow(clk/Kd, n))) + 
             alpha2 * (1 / (1 + pow(not_a/Kd, n))) - delta1 * a)
    
    dnot_a_dt = (alpha1 * (1 / (1 + pow(d/Kd, n) + pow(clk/Kd, n) + 
                 pow(d/Kd, n)*pow(clk/Kd, n))) + 
                 alpha2 * (1 / (1 + pow(a/Kd, n))) - delta1 * not_a)
    
    dq_dt = (alpha3 * ((pow(a/Kd, n)*pow(clk/Kd, n)) / 
             (1 + pow(a/Kd, n) + pow(clk/Kd, n) + pow(a/Kd, n)*pow(clk/Kd, n))) + 
             alpha4 * (1 / (1 + pow(not_q/Kd, n))) - delta2 * q)
    
    dnot_q_dt = (alpha3 * ((pow(not_a/Kd, n)*pow(clk/Kd, n)) / 
                 (1 + pow(not_a/Kd, n) + pow(clk/Kd, n) + 
                 pow(not_a/Kd, n)*pow(clk/Kd, n))) + 
                 alpha4 * (1 / (1 + pow(q/Kd, n))) - delta2 * not_q)

    return np.array([da_dt, dnot_a_dt, dq_dt, dnot_q_dt])

# ==================== PIPO REGISTER CLASS ====================
class PIPORegister:
    """4-bit Parallel In Parallel Out Register with write controls"""
    
    def __init__(self, data_input_func=None):
        self.current_index = 0
        self.switch = False
        self.write_all_state = 0
        self.write_bit_states = [0, 0, 0, 0]
        # Allow external data input function
        self.data_input_func = data_input_func if data_input_func else self._default_parallel_input
        
    def _check_clock_edge(self, T):
        """Detect rising clock edge"""
        clk = get_clock(T)
        rise = get_clock(T - 0.1) < clk
        
        if (clk > 50 and rise) and self.switch:
            self.switch = False
            self.current_index += 1
        if clk < 50 and not self.switch:
            self.switch = True
            
        return clk
    
    def _get_write_signals(self, T):
        """Determine write enable signals based on time windows"""
        # WRITE_ALL windows
        write_all_windows = [(0, 100), (300, 400)]
        write_all = any(start <= T < end for start, end in write_all_windows)
        
        # WRITE_i windows (individual bit control)
        write_bit_windows = {
            1: [(150, 250)],  # Write bit 1 during this window
        }
        
        write_bits = [False] * 4
        for i in range(4):
            if i in write_bit_windows:
                write_bits[i] = any(start <= T < end 
                                   for start, end in write_bit_windows[i])
        
        return write_all, write_bits
    
    def _default_parallel_input(self, T):
        """Default parallel input data as function of time"""
        # Example: different patterns in different time regions
        if T < 100:
            return [100, 0, 100, 0]  # Pattern 1: 1010
        elif 150 <= T < 200:
            return [0, 100, 0, 0]    # Write bit 1 high
        elif 200 <= T < 250:
            return [0, 0, 0, 0]      # Write bit 0 low
        elif 300 <= T < 400:
            return [100, 100, 0, 100]  # Pattern 2: 1101
        else:
            return [0, 0, 0, 0]
    
    def _get_parallel_input(self, T):
        """Get parallel input data - uses external function if provided"""
        return self.data_input_func(T)
    
    def model(self, Y, T, params):
        """PIPO register ODE model"""
        # Unpack state variables for 4 flip-flops
        a1, not_a1, q1, not_q1 = Y[0:4]
        a2, not_a2, q2, not_q2 = Y[4:8]
        a3, not_a3, q3, not_q3 = Y[8:12]
        a4, not_a4, q4, not_q4 = Y[12:16]
        
        # Get clock signal
        clk = self._check_clock_edge(T)
        
        # Get write enable signals
        write_all, write_bits = self._get_write_signals(T)
        
        # Get parallel input data
        parallel_input = self._get_parallel_input(T)
        
        # For each flip-flop, determine D input based on write enables
        q_outputs = [q1, q2, q3, q4]
        d_inputs = []
        
        for i in range(4):
            # If write enabled (either write_all or write_i), use input data
            # Otherwise, hold current value (feedback Q to D)
            if write_all or write_bits[i]:
                d_inputs.append(parallel_input[i])
            else:
                d_inputs.append(q_outputs[i])  # Hold mode
        
        # Build state vectors for each flip-flop
        Y_FF1 = [a1, not_a1, q1, not_q1, d_inputs[0], clk]
        Y_FF2 = [a2, not_a2, q2, not_q2, d_inputs[1], clk]
        Y_FF3 = [a3, not_a3, q3, not_q3, d_inputs[2], clk]
        Y_FF4 = [a4, not_a4, q4, not_q4, d_inputs[3], clk]
        
        # Calculate derivatives for each flip-flop
        dY1 = ff_ode_model(Y_FF1, T, params)
        dY2 = ff_ode_model(Y_FF2, T, params)
        dY3 = ff_ode_model(Y_FF3, T, params)
        dY4 = ff_ode_model(Y_FF4, T, params)
        
        # Concatenate all derivatives
        dY = np.concatenate([dY1, dY2, dY3, dY4])
        
        return dY


# ==================== SIMULATION ====================
if __name__ == "__main__":
    alpha1 = 90.0   # Production rate
    alpha2 = 20.0   # Production rate
    alpha3 = 90.0   # Production rate
    alpha4 = 20.0   # Production rate
    delta1 = 1.23   # Degradation rate
    delta2 = 0.30   # Degradation rate
    Kd = 7.46       # Dissociation constant
    n = 4.0         # Hill coefficient
    
    params = [alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n]
    
    # ========== DEFINE YOUR DATA INPUTS HERE ==========
    # Option 1: Simple function approach
    def my_data_input(t):
        """
        Define what data (D0, D1, D2, D3) you want to write at each time.
        Return [D0, D1, D2, D3] where each value is 0-100
        100 = logic HIGH (1), 0 = logic LOW (0)
        
        Array format: [D0, D1, D2, D3]
                      Bit0 Bit1 Bit2 Bit3
        """
        if t < 100:
            # Binary: Bit3 Bit2 Bit1 Bit0 = 0101
            # Array:  [D0,  D1,  D2,  D3] = [100, 0, 100, 0]
            return [100, 0, 100, 0]
        elif 150 <= t < 200:
            # Only matters for bits being written (WRITE_1 active)
            # Writing Bit1 (D1) = HIGH
            return [0, 100, 0, 0]
        elif 200 <= t < 250:
            # Only matters for bits being written (WRITE_0 active)
            # Writing Bit0 (D0) = LOW
            return [0, 0, 0, 0]
        elif 300 <= t < 400:
            # Binary: Bit3 Bit2 Bit1 Bit0 = 1101
            # Array:  [D0,  D1,  D2,  D3] = [100, 100, 0, 100]
            return [100, 100, 0, 100]
        else:
            return [0, 0, 0, 0]
    
    # Option 2: You can also use a dictionary/lookup table
    # data_patterns = {
    #     (0, 100): [100, 0, 100, 0],      # Pattern A: 1010
    #     (150, 200): [0, 100, 0, 0],      # Pattern B: 0100
    #     (200, 250): [0, 0, 0, 0],        # Pattern C: 0000
    #     (300, 400): [100, 100, 0, 100],  # Pattern D: 1101
    # }
    # 
    # def my_data_input(t):
    #     for (start, end), pattern in data_patterns.items():
    #         if start <= t < end:
    #             return pattern
    #     return [0, 0, 0, 0]  # Default
    
    # ========== CREATE REGISTER WITH YOUR DATA ==========
    register = PIPORegister(data_input_func=my_data_input)
    
    # Initial conditions: 4 flip-flops, each with [a, not_a, q, not_q]
    # Start with all outputs low: q=0, not_q=50
    Y0 = np.array([0, 50, 0, 50] * 4)
    
    # Time span
    t_start = 0
    t_end = 500
    T = np.linspace(t_start, t_end, 5000)
    
    # Solve ODEs
    solution = odeint(register.model, Y0, T, args=(params,))
    
    # Extract Q and not_Q for each bit
    Q = np.column_stack([solution[:, 4*i + 2] for i in range(4)])
    not_Q = np.column_stack([solution[:, 4*i + 3] for i in range(4)])
    
    # Digital output (1 if Q > not_Q, else 0)
    digital_out = (Q > not_Q).astype(int)
    
    # Get control signals for plotting
    clk_signal = np.array([get_clock(t) for t in T])
    write_all_signal = np.zeros(len(T))
    write_bit_signals = np.zeros((len(T), 4))
    
    for idx, t in enumerate(T):
        write_all, write_bits = register._get_write_signals(t)
        write_all_signal[idx] = 50 if write_all else 0
        for i in range(4):
            write_bit_signals[idx, i] = 50 if write_bits[i] else 0
    
    # Plotting - Focus on Bit0 only
    fig, axes = plt.subplots(6, 1, figsize=(14, 12))
    
    # Plot 1: Control signals
    axes[0].plot(T, clk_signal, 'gray', alpha=0.3, label='CLK')
    axes[0].plot(T, write_all_signal, 'r', linewidth=2, label='WRITE_ALL')
    for i in range(4):
        if np.any(write_bit_signals[:, i] > 0):
            axes[0].plot(T, write_bit_signals[:, i], linewidth=2, label=f'WRITE_{i}')
    axes[0].set_ylabel('Control', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('4-bit PIPO Register Simulation', fontsize=14, fontweight='bold')
    axes[0].set_ylim(-5, 105)
    
    # Plot 2-5: Each bit's analog signal (Q only)
    colors = ['blue', 'green', 'orange', 'purple']
    for i in range(4):
        axes[i+1].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
        if np.any(write_bit_signals[:, i] > 0):
            axes[i+1].fill_between(T, 0, 160, 
                                   where=write_bit_signals[:, i] > 0,
                                   alpha=0.15, color='red', 
                                   label=f'WRITE_{i}')
        # Add threshold line
        axes[i+1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        axes[i+1].set_ylabel(f'Bit {i}', fontsize=12)
        axes[i+1].legend(loc='upper right', fontsize=10)
        axes[i+1].grid(True, alpha=0.3)
        axes[i+1].set_ylim(-5, 160)
    
    # Plot 6: Digital outputs (stacked for visibility)
    offset = 0.0
    for i in range(4):
        axes[5].plot(T, digital_out[:, i] + offset, colors[i], 
                    linewidth=2, label=f'Bit{i}')
        axes[5].fill_between(T, offset, 1 + offset, 
                            where=digital_out[:, i] > 0.5,
                            alpha=0.3, color=colors[i])
        offset += 1.5
    axes[5].set_ylabel('Digital Out', fontsize=12)
    axes[5].set_xlabel('Time', fontsize=12)
    axes[5].set_ylim(-0.5, 5)
    axes[5].set_yticks([0, 1.5, 3.0, 4.5])
    axes[5].set_yticklabels(['Bit0', 'Bit1', 'Bit2', 'Bit3'])
    axes[5].legend(loc='upper right', fontsize=10)
    axes[5].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()