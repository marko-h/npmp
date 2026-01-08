import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==================== CONFIGURATION ====================
FLIP_BIT1_0_85 = True  # Enable/disable error injection on bit 1 between t=0-85
FLIP_BIT3_25_125 = True  # Enable/disable error injection on bit 3 between t=25-125
FLIP_STORED_PARITY_VAL = True  # Enable/disable forcing stored parity to 0 between t=400-450
# ==================== PROJECT OVERVIEW ====================
# This module implements a 4-bit PIPO (Parallel In, Parallel Out) register
# using biological D flip-flops. The key feature is PARITY CHECKING support
# for improving reliability in biological systems.
# 
# Architecture:
#   - 4 D flip-flops for data storage (bit 0-3)
#   - 1 D flip-flop for parity bit (5 bits total)
#   - 2 D flip-flops for expected values (q1_expected, q3_expected) - visualization references
#   - Parallel input/output: all 4 bits can be read/written simultaneously
#   - Write control signals: WRITE_ALL (load all bits) and WRITE_i (load individual bits)
#   - Clock-driven state updates following biological reaction kinetics
# 
# Biological Implementation:
#   - Based on synthetic biology with transcription/translation dynamics
#   - Uses Hill functions to model protein-DNA binding cooperativity
#   - ODE system models continuous biochemical state (protein concentrations)
#   - Digital readout: (Q > not_Q) indicates logical state
#
# Parity Checking (Implemented):
#   - For 4 data bits, stores 1 even parity bit (5 bits total storage)
#   - Parity generator: XOR cascade of the 4 data bits
#   - Parity checker: Compare stored parity with recalculated parity
#   - Enables error detection for single-bit errors
#
# Error Injection & Visualization:
#   - Bit 1 (q1) can receive error injection - forced HIGH t=0-100
#   - Bit 1 expected value (q1_expected) tracks what bit 1 should be without error
#   - Bit 3 (q3) can receive error injection - forced HIGH t=25-125, recovery t=125-300
#   - Bit 3 expected value (q3_expected) tracks what bit 3 should be without error
#   - Expected values are used for parity calculation and visualization comparison
#   - This allows visualizing the difference between actual and expected values

# ==================== HILL FUNCTIONS ====================
def get_clock(t, amp=100, per=24, phase=0):
    """
    Clock generator - produces periodic signal for synchronization.
    
    Args:
        t: time (normalized)
        amp: amplitude (0-100 range)
        per: period of oscillation
        phase: phase shift
    
    Returns:
        clock signal value (0 to amp)
    
    Biological Interpretation:
        Represents a periodic input that triggers state transitions
        (e.g., periodic expression of synchronized genes)
    """
    return amp * (np.sin(2 * np.pi * t / per + phase) + 1) / 2

def activate_1(A, Kd, n):
    """
    Hill activation function - YES gate (1 input activation).
    
    Models cooperative binding of activator protein A to DNA.
    Used in synthetic biology to create logic gates.
    
    Args:
        A: activator concentration
        Kd: dissociation constant (binding affinity threshold)
        n: Hill coefficient (cooperativity; n>1 = positive cooperativity)
    
    Returns:
        Normalized output (0 to 1)
        - Low A → output ≈ 0 (gene OFF)
        - High A → output ≈ 1 (gene ON)
    
    Mathematical Form: f(A) = (A/Kd)^n / (1 + (A/Kd)^n)
    """
    return pow(A/Kd, n) / (1 + pow(A/Kd, n))

def repress_1(R, Kd, n):
    """
    Hill repression function - NOT gate (1 input inhibition).
    
    Models cooperative binding of repressor protein R to DNA.
    When R is high, output is low (gene is repressed).
    
    Args:
        R: repressor concentration
        Kd: dissociation constant
        n: Hill coefficient
    
    Returns:
        Normalized output (0 to 1)
        - Low R → output ≈ 1 (gene ON)
        - High R → output ≈ 0 (gene OFF)
    
    Mathematical Form: f(R) = 1 / (1 + (R/Kd)^n)
    """
    return 1 / (1 + pow(R/Kd, n))

# ==================== FLIP-FLOP MODEL ====================
def ff_ode_model(Y, T, params):
    """
    D Flip-Flop ODE Model - Master-Slave SR (Set-Reset) Latch Implementation.
    
    This is the core biological flip-flop circuit. It models a single bit storage
    element using protein concentration dynamics.
    
    STATE VARIABLES (Y = [a, not_a, q, not_q, d, clk]):
        a:      intermediate node "A" (high when input D will be sampled)
        not_a:  complementary state to 'a'
        q:      main output Q (the stored bit value)
        not_q:  complementary output (NOT Q)
        d:      data input signal
        clk:    clock signal (triggers state transitions)
    
    BIOLOGICAL INTERPRETATION:
        Each variable represents protein concentration (e.g., GFP, LacI, etc.)
        The circuit implements logic through:
        - Activation: protein A promotes production of protein B
        - Repression: protein A inhibits production of protein B
        - Competition: AND logic through multiplicative Hill terms
    
    CIRCUIT TOPOLOGY (SR Latch):
        The flip-flop contains two cross-coupled NOR gates forming an SR latch:
        - Set input → forces Q=1 (high)
        - Reset input → forces Q=0 (low)
        - No input → holds current state
    
    PARAMETERS (params = [alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n]):
        alpha1, alpha2: transcription rates for intermediate latch states
        alpha3, alpha4: transcription rates for output Q, not_Q
        delta1, delta2: protein degradation rates
        Kd: dissociation constant (sensitivity threshold)
        n: Hill coefficient (sharpness of logic gate transitions)
    
    The ODE system models:
        da/dt     = production_of_a (depends on D,CLK) - degradation_of_a
        dnot_a/dt = production_of_not_a (depends on a,D,CLK) - degradation_of_not_a
        dq/dt     = production_of_q (depends on a,CLK) - degradation_of_q
        dnot_q/dt = production_of_not_q (depends on not_a,CLK) - degradation_of_not_q
    """
    # Unpack state variables
    a, not_a, q, not_q, d, clk = Y
    alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n = params

    # ===== INTERMEDIATE NODE 'A' DYNAMICS (D input latch) =====
    # a is SET by D (when CLK is high), RESET by not_a (self-repression)
    # This captures the input during clock high phase
    da_dt = (alpha2 * (pow(d/Kd, n) / (1 + pow(d/Kd, n) + pow(clk/Kd, n) + 
             pow(d/Kd, n)*pow(clk/Kd, n))) +  # D and CLK are required (AND logic)
             alpha2 * (1 / (1 + pow(not_a/Kd, n))) -  # Repressed by not_a
             delta1 * a)
    
    # not_a is the complement of 'a' - SET by not_a itself (self-activation)
    # and RESET by a (cross-coupling)
    dnot_a_dt = (alpha1 * (1 / (1 + pow(d/Kd, n) + pow(clk/Kd, n) + 
                 pow(d/Kd, n)*pow(clk/Kd, n))) +  # Repressed by (D AND CLK)
                 alpha2 * (1 / (1 + pow(a/Kd, n))) -  # Repressed by a
                 delta1 * not_a)

    # ===== OUTPUT NODE 'Q' DYNAMICS (Data storage latch) =====
    # q is SET by a (when CLK is high), RESET by not_q (self-repression)
    # This stage samples 'a' on clock rising edge and stores the bit
    dq_dt = (alpha3 * ((pow(a/Kd, n)*pow(clk/Kd, n)) / 
             (1 + pow(a/Kd, n) + pow(clk/Kd, n) + pow(a/Kd, n)*pow(clk/Kd, n))) +  # (a AND CLK)
             alpha4 * (1 / (1 + pow(not_q/Kd, n))) -  # Repressed by not_q
             delta2 * q)
    
    # not_q is complement of 'q' - SET by not_a (when CLK is high), RESET by q
    dnot_q_dt = (alpha3 * ((pow(not_a/Kd, n)*pow(clk/Kd, n)) / 
                 (1 + pow(not_a/Kd, n) + pow(clk/Kd, n) + 
                 pow(not_a/Kd, n)*pow(clk/Kd, n))) +  # (not_a AND CLK)
                 alpha4 * (1 / (1 + pow(q/Kd, n))) -  # Repressed by q
                 delta2 * not_q)

    return np.array([da_dt, dnot_a_dt, dq_dt, dnot_q_dt])

# ==================== PIPO REGISTER CLASS ====================
class PIPORegister:
    """
    4-bit PIPO Register with Parity Checking and Error Injection Capability.
    
    REGISTER FUNCTION:
        - Stores 4 data bits (bit 0-3) + 1 parity bit
        - Tracks expected values (q1_expected, q3_expected) for visualization and parity
        - All bits can be written simultaneously (WRITE_ALL) or individually (WRITE_i)
        - All bits can be read simultaneously from Q outputs
        - Clock-synchronized operation ensures reliable state capture
        - Error injection on bit 1 (q1) and bit 3 (q3) for testing parity detection
    
    WRITE MODES:
        1. WRITE_ALL: Load all 4 bits simultaneously during specified time windows
        2. WRITE_i: Load specific bit 'i' only, during its designated window
        3. WRITE_1ERR, WRITE_3ERR: Error injection signals (when enabled)
        4. HOLD: When no write signal, output Q feeds back to input D (retains value)
    
    ARCHITECTURE:
        - 4 D flip-flops for data bits (bit 0-3)
        - 1 D flip-flop for parity bit
        - 2 D flip-flops for expected values (q1_expected, q3_expected) - visualization only
        - Each flip-flop: [a, not_a, q, not_q] (4 ODEs per bit)
        - Total state vector: 28 variables (7 flip-flops × 4 states)
        - Multiplexer logic: selects D_input or Q_feedback based on write enables
    
    KEY ATTRIBUTES:
        - current_index: tracks edge counter (for debugging)
        - write_all_state, write_bit_states: track current write enable signals
        - data_input_func: user-provided function to specify input data patterns
    """
    
    def __init__(self, data_input_func=None):
        """
        Initialize PIPO register.
        
        Args:
            data_input_func: Optional function that returns input data [D0,D1,D2,D3]
                           as function of time. If None, uses default pattern.
        """
        self.current_index = 0
        self.switch = False
        self.write_all_state = 0
        self.write_bit_states = [0, 0, 0, 0]
        # Allow external data input function
        self.data_input_func = data_input_func if data_input_func else self._default_parallel_input
        
    def _check_clock_edge(self, T):
        """
        Detect rising clock edge for edge counter (diagnostic use only).
        
        Note: The actual register behavior is driven by the continuous
        clock signal in ff_ode_model; this is mainly for tracking transitions.
        
        Args:
            T: current time
        
        Returns:
            clock signal value
        """
        clk = get_clock(T)
        rise = get_clock(T - 0.1) < clk
        
        if (clk > 50 and rise) and self.switch:
            self.switch = False
            self.current_index += 1
        if clk < 50 and not self.switch:
            self.switch = True
            
        return clk
    
    def _get_write_signals(self, T):
        """
        Determine write enable signals based on time windows.
        
        WRITE_ALL windows: All 4 bits are written together
        WRITE_i windows: Only bit 'i' is written (others hold their value)
        WRITE_PARITY: Force write to parity bit (for error injection)
        
        Returns:
            (write_all, write_bits, write_parity)
            - write_all: bool - True if in WRITE_ALL window
            - write_bits: list of 4 bools - True if each bit should be written
            - write_parity: bool - True if parity should be force-written
        """
        # WRITE_ALL windows - simultaneous load of all 4 bits
        # These are 'test' or 'initialization' periods
        write_all_windows = [(0, 100), (300, 400)]
        write_all = any(start <= T < end for start, end in write_all_windows)
        
        # WRITE_i windows - selective individual bit loading
        # Allows testing of specific bits in isolation
        write_bit_windows = {
            1: [(150, 250)],  # Write bit 1 during this window 
        }

        # Conditionally add WRITE_3 error injection window
        if FLIP_BIT3_25_125:
            write_bit_windows[3] = [(25, 125)]  # WRITE_3ERR: Error injection on bit 3
        
        write_bits = [False] * 4
        for i in range(4):
            if i in write_bit_windows:
                write_bits[i] = any(start <= T < end 
                                   for start, end in write_bit_windows[i])
        
        # WRITE_PARITY window - force parity to specific value (error injection)
        write_parity = False
        if FLIP_STORED_PARITY_VAL:
            write_parity = (400 <= T < 450) or (450 <= T < 500)
        
        return write_all, write_bits, write_parity
    
    def _default_parallel_input(self, T):
        """
        Default parallel input data as function of time.
        
        Provides test patterns to demonstrate register operation:
        - Pattern 1 (t=0-100): 1010 binary (bit0=HIGH, bit1=LOW, bit2=HIGH, bit3=LOW)
        - Pattern 2 (t=150-200): Selective write of bit 1 HIGH
        - Pattern 3 (t=200-250): Selective write of bit 0 LOW
        - Pattern 4 (t=300-400): 1101 binary
        
        Returns:
            [D0, D1, D2, D3] where 100=HIGH (logic 1), 0=LOW (logic 0)
        """
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
        """
        Get parallel input data - uses external function if provided.
        
        This method allows users to customize the input patterns without
        modifying the core register code.
        """
        return self.data_input_func(T)
    
    def model(self, Y, T, params):
        """
        PIPO Register ODE Model - Main Simulation Kernel.
        
        This method defines the complete state evolution for all 7 flip-flops.
        Called by the ODE solver at each time step.
        
        STATE LAYOUT (Y array of 28 elements):
            FF1: Y[0:4]   = [a1, not_a1, q1, not_q1]                     (Bit 0)
            FF2: Y[4:8]   = [a2, not_a2, q2, not_q2]                     (Bit 1 - q1)
            FF3: Y[8:12]  = [a3, not_a3, q3, not_q3]                     (Bit 2)
            FF4: Y[12:16] = [a4, not_a4, q4, not_q4]                     (Bit 3 - q3)
            FF5: Y[16:20] = [ap, not_ap, qp, not_qp]                     (Parity bit)
            FF6: Y[20:24] = [a1_expected, not_a1_expected, q1_expected, not_q1_expected] (Bit 1 expected - visualization)
            FF7: Y[24:28] = [a3_expected, not_a3_expected, q3_expected, not_q3_expected] (Bit 3 expected - visualization)
        
        CONTROL FLOW:
            1. Get clock signal (synchronized to time T)
            2. Determine which bits should be written (WRITE_ALL or WRITE_i)
            3. Get parallel input data for this time point
            4. For each flip-flop:
               - If write enabled: set D input = parallel input data
               - If write disabled: set D input = Q output (hold mode)
            5. Apply error injection to bit 1 (q1) if enabled (t=0-100)
            6. Apply error injection to bit 3 (q3) if enabled (t=25-125)
            7. Force bit 3 (q3) recovery after error (t=125-300)
            8. Calculate parity from expected values (q1_expected, q3_expected)
            9. Solve ODE for each flip-flop independently
            10. Concatenate all derivatives and return
        
        Args:
            Y: state vector [28 elements for 7 FF]
            T: current time
            params: [alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n]
        
        Returns:
            dY: derivative of state vector (fed to ODE solver)
        """
        # Unpack state variables for 4 data flip-flops
        a1, not_a1, q1, not_q1 = Y[0:4]   # Bit 0
        a2, not_a2, q2, not_q2 = Y[4:8]   # Bit 1 (q1) - with error injection
        a3, not_a3, q3, not_q3 = Y[8:12]  # Bit 2
        a4, not_a4, q4, not_q4 = Y[12:16] # Bit 3 (q3) - with error injection
        
        # Parity flip-flop state (5th FF): stores even parity of 4 bits
        # Positions 16:20 → [ap, not_ap, qp, not_qp]
        ap, not_ap, qp, not_qp = Y[16:20]
        
        # ===== EXPECTED VALUES (6th & 7th FF) - VISUALIZATION ONLY =====
        # These flip-flops track what bits should be without error injection
        # Used for parity calculation and visualization comparison
        # Bit 1 expected value (6th FF) - Positions 20:24
        a1_expected, not_a1_expected, q1_expected, not_q1_expected = Y[20:24]
        # Bit 3 expected value (7th FF) - Positions 24:28
        a3_expected, not_a3_expected, q3_expected, not_q3_expected = Y[24:28]
        
        # Get clock signal
        clk = self._check_clock_edge(T)
        
        # Get write enable signals
        write_all, write_bits, write_parity = self._get_write_signals(T)
        
        # Get parallel input data
        parallel_input = self._get_parallel_input(T)
        
        # ===== MULTIPLEXER LOGIC =====
        q_outputs = [q1, q2, q3, q4]
        d_inputs = []
        
        for i in range(4):
            if write_all or write_bits[i]:
                d_inputs.append(parallel_input[i])
            else:
                d_inputs.append(q_outputs[i])
        
        # ===== ERROR INJECTION - ACTUAL BIT VALUES =====
        # Apply error injection: force bit 1 (q1) HIGH during t=0-85
        if FLIP_BIT1_0_85 and 0 <= T < 85:
            d_inputs[1] = 100  # Force HIGH (incorrect value)
        # Force bit 1 (q1) back to expected value after error injection ends
        elif FLIP_BIT1_0_85 and 85 <= T < 150 and not write_all and not write_bits[1]:
            d_inputs[1] = 0  # Force back to LOW (expected value)
        
        # Apply error injection: force bit 3 (q3) HIGH during WRITE_3ERR (t=25-125)
        if FLIP_BIT3_25_125 and write_bits[3]:
            d_inputs[3] = 100
        # Force bit 3 (q3) back to expected value after error injection ends
        elif FLIP_BIT3_25_125 and 125 <= T < 300 and not write_all and not write_bits[3]:
            d_inputs[3] = 0  # Force back to LOW (expected value)
        
        # ===== EXPECTED VALUES (6th & 7th FF) - VISUALIZATION REFERENCES =====
        # These flip-flops follow the same logic as their corresponding bits
        # but WITHOUT error injection signals - used for visualization comparison
        
        # Bit 1 expected value (6th FF) - ignores error injection, follows WRITE_ALL and WRITE_1
        if write_all or write_bits[1]:
            d_input_bit1_expected = parallel_input[1]
        else:
            d_input_bit1_expected = q1_expected
        
        # Bit 3 expected value (7th FF) - ignores WRITE_3ERR signal
        if write_all:
            d_input_bit3_expected = parallel_input[3]
        else:
            d_input_bit3_expected = q3_expected
        
        # Calculate parity from expected values (without errors)
        bit0 = 1 if d_inputs[0] >= 50 else 0
        bit1_expected = 1 if d_input_bit1_expected >= 50 else 0
        bit2 = 1 if d_inputs[2] >= 50 else 0
        bit3_expected = 1 if d_input_bit3_expected >= 50 else 0
        
        calc_parity_bit = bit0 ^ bit1_expected ^ bit2 ^ bit3_expected
        d_parity = 100 if calc_parity_bit == 1 else 0
        
        # Apply parity error injection: force stored parity during error windows
        if FLIP_STORED_PARITY_VAL and write_parity:
            if 400 <= T < 450:
                d_parity = 0  # Force parity to 0 (incorrect value)
            elif 450 <= T < 500:
                d_parity = 100  # Force parity back to 1 (recovery)

        # Parity write enable: whenever any bit is written or WRITE_ALL is active or write_parity
        parity_write_enable = write_all or any(write_bits) or write_parity
        d_parity_eff = d_parity if parity_write_enable else qp  # hold when not writing parity

        # ===== BUILD STATE VECTORS FOR ODE SOLVING =====
        # Each flip-flop is independent, so we can solve them separately
        Y_FF1 = [a1, not_a1, q1, not_q1, d_inputs[0], clk]  # Bit 0
        Y_FF2 = [a2, not_a2, q2, not_q2, d_inputs[1], clk]  # Bit 1 (q1) - with error
        Y_FF3 = [a3, not_a3, q3, not_q3, d_inputs[2], clk]  # Bit 2
        Y_FF4 = [a4, not_a4, q4, not_q4, d_inputs[3], clk]  # Bit 3 (q3) - with error
        Y_FFP = [ap, not_ap, qp, not_qp, d_parity_eff, clk] # Parity
        Y_FF1_EXPECTED = [a1_expected, not_a1_expected, q1_expected, not_q1_expected, d_input_bit1_expected, clk]
        Y_FF3_EXPECTED = [a3_expected, not_a3_expected, q3_expected, not_q3_expected, d_input_bit3_expected, clk]
        
        # ===== CALCULATE DERIVATIVES FOR EACH FLIP-FLOP =====
        # Each call to ff_ode_model computes the rate of change
        # for the 4 internal state variables of that flip-flop
        dY1 = ff_ode_model(Y_FF1, T, params)
        dY2 = ff_ode_model(Y_FF2, T, params)
        dY3 = ff_ode_model(Y_FF3, T, params)
        dY4 = ff_ode_model(Y_FF4, T, params)
        dYP = ff_ode_model(Y_FFP, T, params)
        dY1_EXPECTED = ff_ode_model(Y_FF1_EXPECTED, T, params)
        dY3_EXPECTED = ff_ode_model(Y_FF3_EXPECTED, T, params)
        
        # Concatenate all derivatives and return
        dY = np.concatenate([dY1, dY2, dY3, dY4, dYP, dY1_EXPECTED, dY3_EXPECTED])
        
        return dY


# ==================== SIMULATION ====================
if __name__ == "__main__":
    # ===== KINETIC PARAMETERS =====
    # These parameters control the dynamics of the biological system.
    # They represent protein production rates, degradation rates, and binding properties.
    # Tuning these is crucial for achieving robust register operation.
    
    alpha1 = 90.0   # Production rate - controls speed of state transitions
    alpha2 = 20.0   # Production rate - intermediate node dynamics
    alpha3 = 90.0   # Production rate - output Q dynamics
    alpha4 = 20.0   # Production rate - complementary output not_Q
    delta1 = 1.23   # Degradation rate - controls hold time and settling speed
    delta2 = 0.30   # Degradation rate - affects output stability
    Kd = 7.46       # Dissociation constant - protein-DNA binding threshold
    n = 4.0         # Hill coefficient - sharpness of logic transitions (cooperative binding)
    
    params = [alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n]
    
    # ========== DEFINE YOUR DATA INPUTS HERE ==========
    # This is where you customize what data the register stores during simulation.
    # You can define any time-dependent input pattern.
    
    def my_data_input(t):
        """
        Define what data (D0, D1, D2, D3) you want to write at each time.
        
        Return Format: [D0, D1, D2, D3] where:
            - D0: Bit 0 input (100 = logic HIGH/1, 0 = logic LOW/0)
            - D1: Bit 1 input
            - D2: Bit 2 input
            - D3: Bit 3 input
        
        Array indexing: 
            Array position  →  Bit number  →  Binary significance
            [D0,   D1,  D2,  D3] = [Bit0, Bit1, Bit2, Bit3]
        
        Example: [100, 0, 100, 0] = Binary 0101 = Decimal 5 (reading right to left)
        
        Time Windows:
            - t=0-100:     WRITE_ALL active (all 4 bits written simultaneously)
            - t=25-125:    WRITE_3ERR active (only bit 3 written - ERROR INJECTION)
            - t=200-300:   WRITE_1 active (only bit 1 written)
            - t=200-250:   WRITE_0 active (only bit 0 written)
            - t=300-400:   WRITE_ALL active again
        
        Bit 3 Values:
            - t=0-300:     LOW (0) - correct value
            - t=25-125:    HIGH (100) - error injection (only if FLIP_BIT3_25_125=True)
            - t=125-300:   Forced back to LOW (0) - recovery
            - t=300-400:   HIGH (100) - correct value
        """
        if t < 25:
            return [100, 0, 100, 0]
        elif 25 <= t < 125:
            return [100, 0, 100, 0]  # Bit 3 should be LOW
        elif 125 <= t < 150:
            return [0, 0, 0, 0]
        elif 150 <= t < 200:
            return [0, 100, 0, 0]
        elif 200 <= t < 250:
            return [0, 0, 0, 0]
        elif 300 <= t < 400:
            return [100, 100, 0, 100]  # Bit 3 now HIGH
        else:
            return [0, 0, 0, 0]
    
    # ========== CREATE REGISTER WITH YOUR DATA ==========
    # Instantiate the PIPO register with the custom data input function
    register = PIPORegister(data_input_func=my_data_input)
    
    # ========== INITIAL CONDITIONS ==========
    # Y0 specifies the protein concentrations at t=0
    # Format: [a, not_a, q, not_q] for each of 4 flip-flops
    # 
    # Initial state meanings:
    #   q=0, not_q=50 → Output Q is LOW (represents 0)
    #   q=50, not_q=0 → Output Q is HIGH (represents 1)
    # 
    # We start with all outputs LOW (Q=0 for all bits)
    # Include parity flip-flop initial state as LOW as well
    # Include expected value flip-flops (q1_expected, q3_expected) for visualization
    Y0 = np.array([0, 50, 0, 50] * 7)  # 7 FFs (4 data + 1 parity + 2 expected values)
    
    # ========== TIME CONFIGURATION ==========
    # Define the simulation time range and resolution
    t_start = 0      # Start at t=0
    t_end = 500      # Run for 500 time units
    T = np.linspace(t_start, t_end, 5000)  # 5000 time points for smooth curves
    
    # ========== SOLVE ODEs ==========
    # Integrate the system of ODEs using numerical methods (odeint = LSODA)
    # This produces the continuous protein concentration trajectories
    print("Solving differential equations...")
    solution = odeint(register.model, Y0, T, args=(params,))
    print("Solution complete.")
    
    # ========== EXTRACT OUTPUTS ==========
    # The solution array has shape (len(T), 28) → 7 FFs × 4 states
    # Extract the Q and not_Q values for each of the 4 data bits
    Q = np.column_stack([solution[:, 4*i + 2] for i in range(4)])
    not_Q = np.column_stack([solution[:, 4*i + 3] for i in range(4)])
    # Parity stored outputs (5th FF at indices 16..19)
    Qp = solution[:, 16 + 2]
    not_Qp = solution[:, 16 + 3]
    # Expected value outputs (6th & 7th FF) - for visualization
    Q1_expected = solution[:, 20 + 2]  # Bit 1 expected (6th FF at indices 20..23)
    not_Q1_expected = solution[:, 20 + 3]
    Q3_expected = solution[:, 24 + 2]  # Bit 3 expected (7th FF at indices 24..27)
    not_Q3_expected = solution[:, 24 + 3]
    
    # Digital output: Convert analog protein levels to logic levels
    # Digital bit = 1 if Q > not_Q, else 0
    # This threshold comparison is the "biological readout"
    digital_out = (Q > not_Q).astype(int)
    stored_parity = (Qp > not_Qp).astype(int)
    
    # Calculate parity from the bits
    calculated_parity = (digital_out.sum(axis=1) % 2).astype(int)  # even parity bit = XOR of data bits
    parity_error = (stored_parity != calculated_parity).astype(int)
    
    # ========== GET CONTROL SIGNALS FOR VISUALIZATION ==========
    clk_signal = np.array([get_clock(t) for t in T])
    write_all_signal = np.zeros(len(T))
    write_bit_signals = np.zeros((len(T), 4))
    
    for idx, t in enumerate(T):
        write_all, write_bits, write_parity = register._get_write_signals(t)
        write_all_signal[idx] = 50 if write_all else 0
        for i in range(4):
            write_bit_signals[idx, i] = 50 if write_bits[i] else 0
    
    # ========== VISUALIZATION SETUP ==========
    # Create a comprehensive multi-panel figure with 2-column layout:
    # Left/Right arranged for better readability
    # Rows: Controls+Bit0 | Bit1+Bit2 | Bit3+Parity | Digital (bottom)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    ax_flat = axes.flatten()
    
    # ===== SUBPLOT 0 (row 0, left): CONTROL SIGNALS =====
    # Shows when clock oscillates and when write operations are active
    # ===== NEW LAYOUT =====
    # Left column: Control, Parity, Digital Out
    # Right column: Bit 0, 1, 2, 3
    
    # ===== SUBPLOT 0 (row 0, left): CONTROL SIGNALS =====
    ax_flat[0].plot(T, clk_signal, 'gray', alpha=0.3, label='CLK')
    ax_flat[0].plot(T, write_all_signal, 'r', linewidth=2.5, linestyle='--', label='WRITE_ALL')
    for i in range(4):
        if i != 3 and np.any(write_bit_signals[:, i] > 0):
            ax_flat[0].plot(T, write_bit_signals[:, i], linewidth=2.5, linestyle=':', label=f'WRITE_{i}')
    ax_flat[0].set_ylabel('Control', fontsize=12)
    ax_flat[0].legend(loc='upper right', fontsize=9)
    ax_flat[0].grid(True, alpha=0.3)
    ax_flat[0].set_title('4-bit PIPO Register with Even Parity', fontsize=13, fontweight='bold')
    ax_flat[0].set_ylim(-5, 105)
    
    # ===== SUBPLOT 1 (row 0, right): BIT 0 ANALOG OUTPUT =====
    colors = ['blue', 'green', 'orange', 'purple']
    i = 0
    ax_flat[1].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
    if np.any(write_bit_signals[:, i] > 0):
        ax_flat[1].fill_between(T, 0, 400, 
                               where=write_bit_signals[:, i] > 0,
                               alpha=0.15, color='gray', 
                               label=f'WRITE_{i}')
    ax_flat[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_flat[1].set_ylabel(f'Bit {i}', fontsize=12)
    ax_flat[1].legend(loc='upper right', fontsize=9)
    ax_flat[1].grid(True, alpha=0.3)
    ax_flat[1].set_ylim(-5, 400)
    
    # ===== SUBPLOT 2 (row 1, left): PARITY (stored vs calculated) =====
    ax_flat[2].plot(T, stored_parity, 'g', linewidth=2.5, linestyle='--', label='Stored Parity')
    ax_flat[2].plot(T, calculated_parity, 'orange', linewidth=2.5, linestyle=':', label='Calculated Parity')
    ax_flat[2].fill_between(T, 0, 1, where=parity_error > 0.5, alpha=0.25, color='red', label='Error')
    ax_flat[2].set_ylabel('Parity', fontsize=12)
    ax_flat[2].set_ylim(-0.1, 1.1)
    ax_flat[2].legend(loc='lower right', fontsize=9)
    ax_flat[2].grid(True, alpha=0.3)
    ax_flat[2].set_title('Even Parity Check', fontsize=11, fontweight='bold')
    
    # ===== SUBPLOT 3 (row 1, right): BIT 1 ANALOG OUTPUT =====
    i = 1
    if FLIP_BIT1_0_85:
        # Plot actual bit 1 (Q1) as red dotted line - shows value changes
        ax_flat[3].plot(T, Q[:, i], colors[i], linewidth=2, label='Q1 (actual value)')
        # Plot bit 1 expected value (Q1_expected) - what it should be
        ax_flat[3].plot(T, Q1_expected, 'r', linewidth=2.5, linestyle=':', label='Q1 (expected value)')
    else:
        # Plot normal operation throughout (no error injection)
        ax_flat[3].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
    # Show value change region from t=0 to t=100 (error period)
    if FLIP_BIT1_0_85:
        err_region = (T >= 0) & (T < 100)
        ax_flat[3].fill_between(T, 0, 400, 
                               where=err_region,
                               alpha=0.15, color='red', 
                               label='Value Change')
    # Highlight WRITE_1 region using the control signal (aligned to 200-300)
    if np.any(write_bit_signals[:, i] > 0):
        ax_flat[3].fill_between(T, 0, 400,
                               where=write_bit_signals[:, i] > 0,
                               alpha=0.12, color='gray',
                               label='WRITE_1')
    ax_flat[3].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_flat[3].set_ylabel(f'Bit {i}', fontsize=12)
    ax_flat[3].legend(loc='lower right', fontsize=9)
    ax_flat[3].grid(True, alpha=0.3)
    ax_flat[3].set_ylim(-5, 400)
    
    # ===== SUBPLOT 4 (row 2, left): DIGITAL OUTPUT (bits) =====
    offset = 0.0
    for i in range(4):
        ax_flat[4].plot(T, digital_out[:, i] + offset, colors[i], 
                    linewidth=2, label=f'Bit{i}')
        # Fill under the curve to visualize HIGH state more clearly
        ax_flat[4].fill_between(T, offset, 1 + offset, 
                        where=digital_out[:, i] > 0.5,
                        alpha=0.3, color=colors[i])
        offset += 1.5
    ax_flat[4].set_ylabel('Digital Out', fontsize=12)
    ax_flat[4].set_xlabel('Time', fontsize=11)
    ax_flat[4].set_ylim(-0.5, 6.0)
    ax_flat[4].set_yticks([0, 1.5, 3.0, 4.5])
    ax_flat[4].set_yticklabels(['Bit0', 'Bit1', 'Bit2', 'Bit3'])
    ax_flat[4].legend(loc='lower right', fontsize=9)
    ax_flat[4].grid(True, alpha=0.3, axis='x')
    
    # ===== SUBPLOT 5 (row 2, right): BIT 2 ANALOG OUTPUT =====
    i = 2
    ax_flat[5].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
    if np.any(write_bit_signals[:, i] > 0):
        ax_flat[5].fill_between(T, 0, 400, 
                               where=write_bit_signals[:, i] > 0,
                               alpha=0.15, color='gray', 
                               label=f'WRITE_{i}')
    ax_flat[5].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_flat[5].set_ylabel(f'Bit {i}', fontsize=12)
    ax_flat[5].legend(loc='upper right', fontsize=9)
    ax_flat[5].grid(True, alpha=0.3)
    ax_flat[5].set_ylim(-5, 400)
    
    # ===== SUBPLOT 6 (row 3, left): HIDDEN =====
    ax_flat[6].axis('off')
    
    # ===== SUBPLOT 7 (row 3, right): BIT 3 ANALOG OUTPUT =====
    i = 3
    if FLIP_BIT3_25_125:
        # Plot actual bit 3 (q3) as red dotted line - shows value changes
        ax_flat[7].plot(T, Q[:, i], colors[i], linewidth=2, label='Q3 (actual value)')
        # Plot bit 3 expected value (q3_expected) - what it should be
        ax_flat[7].plot(T, Q3_expected, 'r', linewidth=2.5, linestyle=':', label='Q3 (expected value)')
    else:
        # Plot normal operation throughout (no error injection)
        ax_flat[7].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
    # Show value change region from t=25 to t=150 (error + early recovery period)
    if FLIP_BIT3_25_125:
        err_region = (T >= 25) & (T < 150)
        ax_flat[7].fill_between(T, 0, 400, 
                               where=err_region,
                               alpha=0.15, color='red', 
                               label='Value Change')
    ax_flat[7].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_flat[7].set_ylabel(f'Bit {i}', fontsize=12)
    ax_flat[7].legend(loc='lower right', fontsize=9)
    ax_flat[7].grid(True, alpha=0.3)
    ax_flat[7].set_ylim(-5, 400)
    ax_flat[7].set_xlabel('Time', fontsize=11)
    plt.tight_layout()
    plt.show()