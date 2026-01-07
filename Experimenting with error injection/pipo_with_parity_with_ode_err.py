import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==================== PROJECT OVERVIEW ====================
# This module implements a 4-bit PIPO (Parallel In, Parallel Out) register
# using biological D flip-flops. The key feature is PARITY CHECKING support
# for improving reliability in biological systems.
# 
# Architecture:
#   - 4 D flip-flops, each capable of storing one bit
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
# Future Extension (Parity Checking):
#   - For 4 data bits, add 1 parity bit (5 bits total storage)
#   - Parity generator: XOR cascade of the 4 data bits
#   - Parity checker: Compare stored parity with recalculated parity
#   - Enables error detection and improved memory reliability assessment

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

def get_noise_source(t, amp=50, per=47, phase=0):
    """
    Thermal noise source - generates continuous biological noise signal.
    
    Models environmental fluctuations, molecular noise, and stochastic
    biochemical events that can perturb flip-flop states.
    
    Args:
        t: time (normalized)
        amp: noise amplitude (controls error strength)
        per: noise period (different from clock to avoid synchronization)
        phase: phase shift (different for each bit to decorrelate noise)
    
    Returns:
        noise signal value (0 to amp)
    
    Biological Interpretation:
        Represents thermal fluctuations, transcriptional bursting,
        or other stochastic biochemical processes that cause errors.
    """
    # Asymmetric noise pattern to create occasional strong perturbations
    base_noise = np.sin(2 * np.pi * t / per + phase)
    # Add nonlinearity to create bursts
    return amp * (base_noise ** 3 + 1) / 2

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

def get_smooth_error_signal(t, t_start, t_end, strength, ramp_duration=10):
    """
    Smooth error injection signal with gradual ramp-up and ramp-down.
    
    This creates a biologically realistic error profile:
    - Gradual onset (ramp-up): simulates slow degradation or stress accumulation
    - Sustained plateau: error at full strength
    - Gradual recovery (ramp-down): simulates repair mechanisms or stress relief
    
    Args:
        t: current time
        t_start: when error begins (start of ramp-up)
        t_end: when error ends (end of ramp-down)
        strength: maximum error forcing magnitude
        ramp_duration: time for smooth transitions (default 10)
    
    Returns:
        Error signal value (0 to strength)
        
    Biological Interpretation:
        Models gradual environmental perturbations rather than instantaneous shocks.
        Mimics realistic biological processes like:
        - Temperature stress building up and dissipating
        - Gradual resource depletion and replenishment
        - Slow accumulation and clearance of toxic metabolites
    """
    if t < t_start:
        # Before error window
        return 0.0
    elif t < t_start + ramp_duration:
        # Ramp-up phase: smooth sigmoid transition from 0 to strength
        progress = (t - t_start) / ramp_duration
        # Use smooth sigmoid: 3*x^2 - 2*x^3 (smoothstep function)
        smooth_factor = 3 * progress**2 - 2 * progress**3
        return strength * smooth_factor
    elif t < t_end - ramp_duration:
        # Plateau phase: sustained error at full strength
        return strength
    elif t < t_end:
        # Ramp-down phase: smooth sigmoid transition from strength to 0
        progress = (t - (t_end - ramp_duration)) / ramp_duration
        smooth_factor = 3 * progress**2 - 2 * progress**3
        return strength * (1 - smooth_factor)
    else:
        # After error window
        return 0.0

# ==================== FLIP-FLOP MODEL ==
def ff_ode_model(Y, T, params, noise=0, noise_strength=0, error_forcing=0):
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
    
    NOISE PARAMETERS:
        noise: continuous noise signal (0-100) from thermal fluctuations
        noise_strength: scaling factor for noise influence on flip-flop
        error_forcing: biological error injection (>0 forces HIGH, <0 forces LOW)
    
    The ODE system models:
        da/dt     = production_of_a (depends on D,CLK) - degradation_of_a
        dnot_a/dt = production_of_not_a (depends on a,D,CLK) - degradation_of_not_a
        dq/dt     = production_of_q (depends on a,CLK) - degradation_of_q + NOISE + ERROR_FORCING
        dnot_q/dt = production_of_not_q (depends on not_a,CLK) - degradation_of_not_q + NOISE - ERROR_FORCING
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
    # NOISE TERM: When noise is high AND q is near threshold (unstable), adds perturbation
    noise_activation_q = activate_1(noise, Kd, n)  # Hill activation by noise
    stability_q = abs(q - not_q)  # How stable is current state?
    noise_sensitivity_q = repress_1(stability_q, 30, 2)  # More sensitive when unstable
    noise_effect_q = noise_strength * noise_activation_q * noise_sensitivity_q * (not_q - q) / 100
    
    # ERROR FORCING: Biological error injection - adds/removes protein production
    # Positive forcing → drives Q HIGH (increases q, decreases not_q)
    # Negative forcing → drives Q LOW (decreases q, increases not_q)
    # This simulates environmental perturbations or targeted mutations
    
    dq_dt = (alpha3 * ((pow(a/Kd, n)*pow(clk/Kd, n)) / 
             (1 + pow(a/Kd, n) + pow(clk/Kd, n) + pow(a/Kd, n)*pow(clk/Kd, n))) +  # (a AND CLK)
             alpha4 * (1 / (1 + pow(not_q/Kd, n))) -  # Repressed by not_q
             delta2 * q +
             noise_effect_q +  # Thermal noise perturbation
             error_forcing)  # Manual error injection (biological forcing)
    
    # not_q is complement of 'q' - SET by not_a (when CLK is high), RESET by q
    # Noise affects not_q in opposite direction to maintain complementary nature
    dnot_q_dt = (alpha3 * ((pow(not_a/Kd, n)*pow(clk/Kd, n)) / 
                 (1 + pow(not_a/Kd, n) + pow(clk/Kd, n) + 
                 pow(not_a/Kd, n)*pow(clk/Kd, n))) +  # (not_a AND CLK)
                 alpha4 * (1 / (1 + pow(q/Kd, n))) -  # Repressed by q
                 delta2 * not_q -
                 noise_effect_q -  # Opposite noise effect
                 error_forcing)  # Opposite error forcing

    return np.array([da_dt, dnot_a_dt, dq_dt, dnot_q_dt])

# ==================== PIPO REGISTER CLASS ====================
class PIPORegister:
    """
    4-bit Parallel In Parallel Out (PIPO) Register with selective write controls.
    
    REGISTER FUNCTION:
        - Stores 4 bits of data in parallel
        - All bits can be written simultaneously (WRITE_ALL) or individually (WRITE_i)
        - All bits can be read simultaneously from Q outputs
        - Clock-synchronized operation ensures reliable state capture
    
    WRITE MODES:
        1. WRITE_ALL: Load all 4 bits simultaneously during specified time windows
        2. WRITE_i: Load specific bit 'i' only, during its designated window
        3. HOLD: When no write signal, output Q feeds back to input D (retains value)
    
    ARCHITECTURE:
        - 4 D flip-flops arranged in parallel
        - Each flip-flop: [a, not_a, q, not_q] (4 ODEs per bit)
        - Total state vector: 16 variables
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
        
        Returns:
            (write_all, write_bits)
            - write_all: bool - True if in WRITE_ALL window
            - write_bits: list of 4 bools - True if each bit should be written
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
        
        write_bits = [False] * 4
        for i in range(4):
            if i in write_bit_windows:
                write_bits[i] = any(start <= T < end 
                                   for start, end in write_bit_windows[i])
        
        return write_all, write_bits
    
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
    
    def model(self, Y, T, params, noise_strength=2.0):
        """
        PIPO Register ODE Model - Main Simulation Kernel.
        
        This method defines the complete state evolution for all 4 flip-flops.
        Called by the ODE solver at each time step.
        
        STATE LAYOUT (Y array of 20 elements):
            FF1: Y[0:4]   = [a1, not_a1, q1, not_q1]
            FF2: Y[4:8]   = [a2, not_a2, q2, not_q2]
            FF3: Y[8:12]  = [a3, not_a3, q3, not_q3]
            FF4: Y[12:16] = [a4, not_a4, q4, not_q4]
        
        CONTROL FLOW:
            1. Get clock signal (synchronized to time T)
            2. Determine which bits should be written (WRITE_ALL or WRITE_i)
            3. Get parallel input data for this time point
            4. Generate noise sources for each bit (different phase for decorrelation)
            5. For each flip-flop:
               - If write enabled: set D input = parallel input data
               - If write disabled: set D input = Q output (hold mode)
            6. Solve ODE for each flip-flop with noise perturbation
            7. Concatenate all derivatives and return
        
        Args:
            Y: state vector [20 elements for 5 FF]
            T: current time
            params: [alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n]
            noise_strength: scaling factor for thermal noise (default 2.0)
        
        Returns:
            dY: derivative of state vector (fed to ODE solver)
        """
        # Unpack state variables for 4 flip-flops
        a1, not_a1, q1, not_q1 = Y[0:4]
        a2, not_a2, q2, not_q2 = Y[4:8]
        a3, not_a3, q3, not_q3 = Y[8:12]
        a4, not_a4, q4, not_q4 = Y[12:16]
        # Parity flip-flop state (5th FF): stores even parity of 4 bits
        # Positions 16:20 → [ap, not_ap, qp, not_qp]
        ap, not_ap, qp, not_qp = Y[16:20]
        
        # Get clock signal
        clk = self._check_clock_edge(T)
        
        # Get write enable signals
        write_all, write_bits = self._get_write_signals(T)
        
        # Get parallel input data
        parallel_input = self._get_parallel_input(T)
        
        # ===== RANDOM NOISE SOURCES (COMMENTED OUT FOR NOW) =====
        # Generate independent noise sources for each bit (different phases for decorrelation)
        # Each bit experiences different thermal fluctuations
        # noise_sources = [
        #     get_noise_source(T, amp=50, per=47, phase=0),      # Bit 0 noise
        #     get_noise_source(T, amp=50, per=53, phase=np.pi/3),  # Bit 1 noise
        #     get_noise_source(T, amp=50, per=59, phase=2*np.pi/3),  # Bit 2 noise
        #     get_noise_source(T, amp=50, per=61, phase=np.pi)   # Bit 3 noise
        # ]
        noise_sources = [0, 0, 0, 0]  # No random noise for now
        
        # ===== MANUAL ERROR FORCING (Biological Error Injection) =====
        # Bit 3 remains INTACT at zero - no error forcing applied
        # This demonstrates error detection capability: we visualize what an error would look like
        # but the actual flip-flop state is preserved
        error_forcing = [0, 0, 0, 0]  # No forcing - all bits intact
        
        # ===== MULTIPLEXER LOGIC =====
        # For each flip-flop, determine D input based on write enables
        # This is the key difference from a simple D flip-flop:
        # - Write enabled → D = input data (new value)
        # - Write disabled → D = Q output (hold current value)
        q_outputs = [q1, q2, q3, q4]
        d_inputs = []
        
        for i in range(4):
            # If write enabled (either write_all or write_i), use input data
            # Otherwise, hold current value (feedback Q to D)
            if write_all or write_bits[i]:
                d_inputs.append(parallel_input[i])
            else:
                d_inputs.append(q_outputs[i])  # Hold mode: feedback Q→D
        
        # Compute desired digital values (0/1) for parity from the d_inputs
        desired_bits_binary = [1 if val >= 50 else 0 for val in d_inputs]
        calc_parity_bit = desired_bits_binary[0] ^ desired_bits_binary[1] ^ desired_bits_binary[2] ^ desired_bits_binary[3]
        d_parity = 100 if calc_parity_bit == 1 else 0

        # Parity write enable: whenever any bit is written or WRITE_ALL is active
        parity_write_enable = write_all or any(write_bits)
        d_parity_eff = d_parity if parity_write_enable else qp  # hold when not writing parity

        # ===== BUILD STATE VECTORS FOR ODE SOLVING =====
        # Each flip-flop is independent, so we can solve them separately
        Y_FF1 = [a1, not_a1, q1, not_q1, d_inputs[0], clk]
        Y_FF2 = [a2, not_a2, q2, not_q2, d_inputs[1], clk]
        Y_FF3 = [a3, not_a3, q3, not_q3, d_inputs[2], clk]
        Y_FF4 = [a4, not_a4, q4, not_q4, d_inputs[3], clk]
        Y_FFP = [ap, not_ap, qp, not_qp, d_parity_eff, clk]
        
        # ===== CALCULATE DERIVATIVES FOR EACH FLIP-FLOP =====
        # Each call to ff_ode_model computes the rate of change
        # for the 4 internal state variables of that flip-flop
        # Noise is applied to each bit independently for realistic thermal effects
        # Error forcing is applied to bit 3 during specific time windows for controlled error injection
        dY1 = ff_ode_model(Y_FF1, T, params, noise=noise_sources[0], noise_strength=noise_strength, error_forcing=error_forcing[0])
        dY2 = ff_ode_model(Y_FF2, T, params, noise=noise_sources[1], noise_strength=noise_strength, error_forcing=error_forcing[1])
        dY3 = ff_ode_model(Y_FF3, T, params, noise=noise_sources[2], noise_strength=noise_strength, error_forcing=error_forcing[2])
        dY4 = ff_ode_model(Y_FF4, T, params, noise=noise_sources[3], noise_strength=noise_strength, error_forcing=error_forcing[3])
        # Parity bit gets no noise or forcing (stored in controlled environment)
        dYP = ff_ode_model(Y_FFP, T, params, noise=0, noise_strength=0, error_forcing=0)
        
        # Concatenate all derivatives and return
        dY = np.concatenate([dY1, dY2, dY3, dY4, dYP])
        
        return dY
    
    def model_with_bit3_error(self, Y, T, params, noise_strength=2.0):
        """
        PIPO Register ODE Model WITH BIT 3 ERROR FORCING.
        Identical to model() but forces bit 3 HIGH during error window for comparison.
        Used to generate hypothetical error trajectory for visualization.
        """
        # Unpack state variables for 4 flip-flops
        a1, not_a1, q1, not_q1 = Y[0:4]
        a2, not_a2, q2, not_q2 = Y[4:8]
        a3, not_a3, q3, not_q3 = Y[8:12]
        a4, not_a4, q4, not_q4 = Y[12:16]
        ap, not_ap, qp, not_qp = Y[16:20]
        
        # Get clock signal
        
        clk = self._check_clock_edge(T)
        
        # Get write enable signals
        write_all, write_bits = self._get_write_signals(T)
        
        # Get parallel input data
        parallel_input = self._get_parallel_input(T)
        
        # No random noise for this calculation
        noise_sources = [0, 0, 0, 0]
        
        # ===== ERROR FORCING FOR BIT 3 (Smooth Biological Perturbation) =====
        # Configure error injection parameters
        ERROR_BIT = 3              # Which bit to perturb
        ERROR_START = 50           # Error window start time
        ERROR_END = 150            # Error window end time
        ERROR_STRENGTH = 30.0      # Maximum forcing magnitude
        ERROR_RAMP = 10            # Ramp duration for smooth transitions
        
        # Calculate smooth error signal with gradual onset and recovery
        error_forcing = [0, 0, 0, 0]
        error_forcing[ERROR_BIT] = get_smooth_error_signal(
            T, ERROR_START, ERROR_END, ERROR_STRENGTH, ERROR_RAMP
        )
        
        # Multiplexer logic
        q_outputs = [q1, q2, q3, q4]
        d_inputs = []
        
        for i in range(4):
            if write_all or write_bits[i]:
                d_inputs.append(parallel_input[i])
            else:
                d_inputs.append(q_outputs[i])
        
        # Parity calculation
        desired_bits_binary = [1 if val >= 50 else 0 for val in d_inputs]
        calc_parity_bit = desired_bits_binary[0] ^ desired_bits_binary[1] ^ desired_bits_binary[2] ^ desired_bits_binary[3]
        d_parity = 100 if calc_parity_bit == 1 else 0
        parity_write_enable = write_all or any(write_bits)
        d_parity_eff = d_parity if parity_write_enable else qp
        
        # Build state vectors
        Y_FF1 = [a1, not_a1, q1, not_q1, d_inputs[0], clk]
        Y_FF2 = [a2, not_a2, q2, not_q2, d_inputs[1], clk]
        Y_FF3 = [a3, not_a3, q3, not_q3, d_inputs[2], clk]
        Y_FF4 = [a4, not_a4, q4, not_q4, d_inputs[3], clk]
        Y_FFP = [ap, not_ap, qp, not_qp, d_parity_eff, clk]
        
        # Calculate derivatives with error forcing
        dY1 = ff_ode_model(Y_FF1, T, params, noise=noise_sources[0], noise_strength=0, error_forcing=error_forcing[0])
        dY2 = ff_ode_model(Y_FF2, T, params, noise=noise_sources[1], noise_strength=0, error_forcing=error_forcing[1])
        dY3 = ff_ode_model(Y_FF3, T, params, noise=noise_sources[2], noise_strength=0, error_forcing=error_forcing[2])
        dY4 = ff_ode_model(Y_FF4, T, params, noise=noise_sources[3], noise_strength=0, error_forcing=error_forcing[3])
        dYP = ff_ode_model(Y_FFP, T, params, noise=0, noise_strength=0, error_forcing=0)
        
        dY = np.concatenate([dY1, dY2, dY3, dY4, dYP])
        return dY
        """
        PIPO Register ODE Model - Main Simulation Kernel.
        
        This method defines the complete state evolution for all 4 flip-flops.
        Called by the ODE solver at each time step.
        
        STATE LAYOUT (Y array of 16 elements):
            FF1: Y[0:4]   = [a1, not_a1, q1, not_q1]
            FF2: Y[4:8]   = [a2, not_a2, q2, not_q2]
            FF3: Y[8:12]  = [a3, not_a3, q3, not_q3]
            FF4: Y[12:16] = [a4, not_a4, q4, not_q4]
        
        CONTROL FLOW:
            1. Get clock signal (synchronized to time T)
            2. Determine which bits should be written (WRITE_ALL or WRITE_i)
            3. Get parallel input data for this time point
            4. Generate noise sources for each bit (different phase for decorrelation)
            5. For each flip-flop:
               - If write enabled: set D input = parallel input data
               - If write disabled: set D input = Q output (hold mode)
            6. Solve ODE for each flip-flop with noise perturbation
            7. Concatenate all derivatives and return
        
        Args:
            Y: state vector [20 elements for 5 FF]
            T: current time
            params: [alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n]
            noise_strength: scaling factor for thermal noise (default 2.0)
        
        Returns:
            dY: derivative of state vector (fed to ODE solver)
        """
        # Unpack state variables for 4 flip-flops
        a1, not_a1, q1, not_q1 = Y[0:4]
        a2, not_a2, q2, not_q2 = Y[4:8]
        a3, not_a3, q3, not_q3 = Y[8:12]
        a4, not_a4, q4, not_q4 = Y[12:16]
        # Parity flip-flop state (5th FF): stores even parity of 4 bits
        # Positions 16:20 → [ap, not_ap, qp, not_qp]
        ap, not_ap, qp, not_qp = Y[16:20]
        
        # Get clock signal
        clk = self._check_clock_edge(T)
        
        # Get write enable signals
        write_all, write_bits = self._get_write_signals(T)
        
        # Get parallel input data
        parallel_input = self._get_parallel_input(T)
        
        # ===== RANDOM NOISE SOURCES (COMMENTED OUT FOR NOW) =====
        # Generate independent noise sources for each bit (different phases for decorrelation)
        # Each bit experiences different thermal fluctuations
        # noise_sources = [
        #     get_noise_source(T, amp=50, per=47, phase=0),      # Bit 0 noise
        #     get_noise_source(T, amp=50, per=53, phase=np.pi/3),  # Bit 1 noise
        #     get_noise_source(T, amp=50, per=59, phase=2*np.pi/3),  # Bit 2 noise
        #     get_noise_source(T, amp=50, per=61, phase=np.pi)   # Bit 3 noise
        # ]
        noise_sources = [0, 0, 0, 0]  # No random noise for now
        
        # ===== MANUAL ERROR FORCING (Biological Error Injection) =====
        # Bit 3 remains INTACT at zero - no error forcing applied
        # This demonstrates error detection capability: we visualize what an error would look like
        # but the actual flip-flop state is preserved
        error_forcing = [0, 0, 0, 0]  # No forcing - all bits intact
        
        # ===== MULTIPLEXER LOGIC =====
        # For each flip-flop, determine D input based on write enables
        # This is the key difference from a simple D flip-flop:
        # - Write enabled → D = input data (new value)
        # - Write disabled → D = Q output (hold current value)
        q_outputs = [q1, q2, q3, q4]
        d_inputs = []
        
        for i in range(4):
            # If write enabled (either write_all or write_i), use input data
            # Otherwise, hold current value (feedback Q to D)
            if write_all or write_bits[i]:
                d_inputs.append(parallel_input[i])
            else:
                d_inputs.append(q_outputs[i])  # Hold mode: feedback Q→D
        
        # Compute desired digital values (0/1) for parity from the d_inputs
        desired_bits_binary = [1 if val >= 50 else 0 for val in d_inputs]
        calc_parity_bit = desired_bits_binary[0] ^ desired_bits_binary[1] ^ desired_bits_binary[2] ^ desired_bits_binary[3]
        d_parity = 100 if calc_parity_bit == 1 else 0

        # Parity write enable: whenever any bit is written or WRITE_ALL is active
        parity_write_enable = write_all or any(write_bits)
        d_parity_eff = d_parity if parity_write_enable else qp  # hold when not writing parity

        # ===== BUILD STATE VECTORS FOR ODE SOLVING =====
        # Each flip-flop is independent, so we can solve them separately
        Y_FF1 = [a1, not_a1, q1, not_q1, d_inputs[0], clk]
        Y_FF2 = [a2, not_a2, q2, not_q2, d_inputs[1], clk]
        Y_FF3 = [a3, not_a3, q3, not_q3, d_inputs[2], clk]
        Y_FF4 = [a4, not_a4, q4, not_q4, d_inputs[3], clk]
        Y_FFP = [ap, not_ap, qp, not_qp, d_parity_eff, clk]
        
        # ===== CALCULATE DERIVATIVES FOR EACH FLIP-FLOP =====
        # Each call to ff_ode_model computes the rate of change
        # for the 4 internal state variables of that flip-flop
        # Noise is applied to each bit independently for realistic thermal effects
        # Error forcing is applied during specific time windows for controlled error injection
        dY1 = ff_ode_model(Y_FF1, T, params, noise=noise_sources[0], noise_strength=noise_strength, error_forcing=error_forcing[0])
        dY2 = ff_ode_model(Y_FF2, T, params, noise=noise_sources[1], noise_strength=noise_strength, error_forcing=error_forcing[1])
        dY3 = ff_ode_model(Y_FF3, T, params, noise=noise_sources[2], noise_strength=noise_strength, error_forcing=error_forcing[2])
        dY4 = ff_ode_model(Y_FF4, T, params, noise=noise_sources[3], noise_strength=noise_strength, error_forcing=error_forcing[3])
        # Parity bit gets no noise or forcing (stored in controlled environment)
        dYP = ff_ode_model(Y_FFP, T, params, noise=0, noise_strength=0, error_forcing=0)
        
        # Concatenate all derivatives and return
        dY = np.concatenate([dY1, dY2, dY3, dY4, dYP])
        
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
            - t=150-250:   WRITE_1 active (only bit 1 written)
            - t=200-250:   WRITE_0 active (only bit 0 written)
            - t=300-400:   WRITE_ALL active again
        """
        if t < 100:
            # Binary: Bit3 Bit2 Bit1 Bit0 = 0101
            # Array:  [D0,  D1,  D2,  D3] = [100, 0, 100, 0]
            # Biological: All 4 flip-flops load this value
            return [100, 0, 100, 0]
        elif 150 <= t < 200:
            # Only bit 1 is being written (WRITE_1 active)
            # Other bits retain their stored values (hold mode)
            return [0, 100, 0, 0]
        elif 200 <= t < 250:
            # Only bit 0 is being written (WRITE_0 active)
            return [0, 0, 0, 0]
        elif 300 <= t < 400:
            # Binary: Bit3 Bit2 Bit1 Bit0 = 1101
            # Array:  [D0,  D1,  D2,  D3] = [100, 100, 0, 100]
            return [100, 100, 0, 100]
        else:
            # No write active - all flip-flops hold their current values
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
    Y0 = np.array([0, 50, 0, 50] * 5)  # 5 FFs (4 data + 1 parity)
    
    # ========== TIME CONFIGURATION ==========
    # Define the simulation time range and resolution
    t_start = 0      # Start at t=0
    t_end = 500      # Run for 500 time units
    T = np.linspace(t_start, t_end, 5000)  # 5000 time points for smooth curves
    
    # ========== ERROR INJECTION CONFIGURATION ==========
    # Configure biological error perturbation for demonstration
    ERROR_CONFIG = {
        'bit_index': 3,           # Which bit to perturb (0-3)
        'start_time': 50,         # When error begins
        'end_time': 150,          # When error ends
        'strength': 30.0,         # Maximum forcing magnitude
        'ramp_duration': 10,      # Time for smooth ramp-up/ramp-down
    }
    
    # ========== SOLVE ODEs ==========
    # Integrate the system of ODEs using numerical methods (odeint = LSODA)
    # This produces the continuous protein concentration trajectories
    # noise_strength controls the amplitude of thermal noise effects (currently disabled)
    # Error forcing uses smooth ramp-up/ramp-down for biological realism
    noise_strength = 0.0  # Disabled for now (was 2.0) - using smooth error injection instead
    print("Solving differential equations...")
    print(f"  Noise strength: {noise_strength} (random noise disabled)")
    print(f"  Error protection: Bit {ERROR_CONFIG['bit_index']} INTACT - no errors allowed")
    print(f"  Visualization: Red dotted line shows hypothetical error ({ERROR_CONFIG['start_time']}-{ERROR_CONFIG['end_time']})")
    print(f"  Error profile: Smooth ramp-up/down over {ERROR_CONFIG['ramp_duration']} time units")
    solution = odeint(register.model, Y0, T, args=(params, noise_strength))
    print("Solution complete.")
    
    # ========== EXTRACT OUTPUTS ==========
    # The solution array has shape (len(T), 20) → 5 FFs × 4 states
    # Extract the Q and not_Q values for each of the 4 data bits
    Q = np.column_stack([solution[:, 4*i + 2] for i in range(4)])
    not_Q = np.column_stack([solution[:, 4*i + 3] for i in range(4)])
    # Parity stored outputs (5th FF at indices 16..19)
    Qp = solution[:, 16 + 2]
    not_Qp = solution[:, 16 + 3]
    
    # ========== CALCULATE HYPOTHETICAL ERROR STATE FOR BIT 3 ==========
    # Solve ODE again with error forcing enabled to show what would happen
    # This uses Hill function dynamics to realistically show error propagation
    print("\nCalculating hypothetical error trajectory (Hill function-based)...")
    solution_with_error = odeint(register.model_with_bit3_error, Y0, T, args=(params, 0.0))
    Q_with_error = np.column_stack([solution_with_error[:, 4*i + 2] for i in range(4)])
    Q_hypothetical_error_3 = Q_with_error[:, 3]  # Extract bit 3 with error forcing
    print("Hypothetical trajectory complete.")
    
    # Digital output: Convert analog protein levels to logic levels
    # Digital bit = 1 if Q > not_Q, else 0
    # This threshold comparison is the "biological readout"
    digital_out = (Q > not_Q).astype(int)
    stored_parity = (Qp > not_Qp).astype(int)
    
    # ========== ERROR ANALYSIS ==========
    # Errors are now generated continuously via Hill function-based noise in the ODE model
    # No need for post-hoc injection - errors emerge naturally from the dynamics
    
    # Recalculate parity from the noisy output
    calculated_parity = (digital_out.sum(axis=1) % 2).astype(int)  # even parity bit = XOR of data bits
    parity_error = (stored_parity != calculated_parity).astype(int)
    
    # Print error statistics
    total_errors = parity_error.sum()
    error_rate = total_errors / len(T) * 100
    print(f"\n{'='*60}")
    print(f"BIOLOGICAL ERROR INJECTION REPORT")
    print(f"{'='*60}")
    for i in range(4):
        bit_flips = np.sum(np.diff(digital_out[:, i]) != 0)
        print(f"  Bit {i}: {bit_flips} state transitions")
    print(f"  Total parity errors detected: {total_errors}")
    print(f"  Error rate: {error_rate:.2f}% of time points")
    print(f"  Status: Bit {ERROR_CONFIG['bit_index']} is PROTECTED - remains at zero")
    print(f"  Error profile: Smooth ramp (±{ERROR_CONFIG['ramp_duration']} time units) at boundaries")
    print(f"  Visualization: Red dotted shows gradual error onset/recovery ({ERROR_CONFIG['start_time']}-{ERROR_CONFIG['end_time']})")
    print(f"{'='*60}\n")
    
    # ========== GET CONTROL SIGNALS FOR VISUALIZATION ==========
    clk_signal = np.array([get_clock(t) for t in T])
    write_all_signal = np.zeros(len(T))
    write_bit_signals = np.zeros((len(T), 4))
    
    for idx, t in enumerate(T):
        write_all, write_bits = register._get_write_signals(t)
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
    ax_flat[0].plot(T, write_all_signal, 'r', linewidth=2, label='WRITE_ALL')
    for i in range(4):
        if np.any(write_bit_signals[:, i] > 0):
            ax_flat[0].plot(T, write_bit_signals[:, i], linewidth=2, label=f'WRITE_{i}')
    ax_flat[0].set_ylabel('Control', fontsize=12)
    ax_flat[0].legend(loc='upper right', fontsize=9)
    ax_flat[0].grid(True, alpha=0.3)
    ax_flat[0].set_title('4-bit PIPO Register with Even Parity', fontsize=13, fontweight='bold')
    ax_flat[0].set_ylim(-5, 105)
    
    # ===== SUBPLOT 1 (row 0, right): BIT 0 ANALOG OUTPUT =====
    colors = ['blue', 'green', 'orange', 'purple']
    i = 0
    ax_flat[1].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
    # Show threshold line
    ax_flat[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')
    if np.any(write_bit_signals[:, i] > 0):
        ax_flat[1].fill_between(T, 0, 400, 
                               where=write_bit_signals[:, i] > 0,
                               alpha=0.15, color='gray', 
                               label=f'WRITE_{i}')
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
    ax_flat[2].legend(loc='upper right', fontsize=9)
    ax_flat[2].grid(True, alpha=0.3)
    ax_flat[2].set_title('Even Parity Check', fontsize=11, fontweight='bold')
    
    # ===== SUBPLOT 3 (row 1, right): BIT 1 ANALOG OUTPUT =====
    i = 1
    ax_flat[3].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
    ax_flat[3].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')
    if np.any(write_bit_signals[:, i] > 0):
        ax_flat[3].fill_between(T, 0, 400, 
                               where=write_bit_signals[:, i] > 0,
                               alpha=0.15, color='gray', 
                               label=f'WRITE_{i}')
    ax_flat[3].set_ylabel(f'Bit {i}', fontsize=12)
    ax_flat[3].legend(loc='upper right', fontsize=9)
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
    ax_flat[4].legend(loc='upper right', fontsize=9)
    ax_flat[4].grid(True, alpha=0.3, axis='x')
    
    # ===== SUBPLOT 5 (row 2, right): BIT 2 ANALOG OUTPUT =====
    i = 2
    ax_flat[5].plot(T, Q[:, i], colors[i], linewidth=2, label=f'Q{i}')
    ax_flat[5].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')
    if np.any(write_bit_signals[:, i] > 0):
        ax_flat[5].fill_between(T, 0, 400, 
                               where=write_bit_signals[:, i] > 0,
                               alpha=0.15, color='gray', 
                               label=f'WRITE_{i}')
    ax_flat[5].set_ylabel(f'Bit {i}', fontsize=12)
    ax_flat[5].legend(loc='upper right', fontsize=9)
    ax_flat[5].grid(True, alpha=0.3)
    ax_flat[5].set_ylim(-5, 400)
    
    # ===== SUBPLOT 6 (row 3, left): HIDDEN =====
    ax_flat[6].axis('off')
    
    # ===== SUBPLOT 7 (row 3, right): BIT 3 ANALOG OUTPUT =====
    i = 3
    # Actual bit 3 state (solid violet line) - remains INTACT at 0
    ax_flat[7].plot(T, Q[:, i], colors[i], linewidth=2.5, label=f'Q{i} (actual - intact)', zorder=3)
    # Hypothetical error state (red dotted line) - shows what WOULD happen if error occurred
    error_mask = (T >= ERROR_CONFIG['start_time']) & (T < ERROR_CONFIG['end_time'])
    if np.any(error_mask):
        ax_flat[7].plot(T[error_mask], Q_hypothetical_error_3[error_mask], 'r:', linewidth=2.5, alpha=0.8, label=f'Q{i} (hypothetical error)', zorder=2)
        # Shade the error injection window (including ramp regions)
        ax_flat[7].axvspan(ERROR_CONFIG['start_time'], ERROR_CONFIG['end_time'], 
                          alpha=0.1, color='red', label='Error window (blocked)', zorder=1)
        # Mark ramp regions with lighter shading
        ramp = ERROR_CONFIG['ramp_duration']
        ax_flat[7].axvspan(ERROR_CONFIG['start_time'], ERROR_CONFIG['start_time'] + ramp,
                          alpha=0.05, color='orange', label='Ramp-up', zorder=0)
        ax_flat[7].axvspan(ERROR_CONFIG['end_time'] - ramp, ERROR_CONFIG['end_time'],
                          alpha=0.05, color='orange', label='Ramp-down', zorder=0)
    ax_flat[7].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')
    if np.any(write_bit_signals[:, i] > 0):
        ax_flat[7].fill_between(T, 0, 400, 
                               where=write_bit_signals[:, i] > 0,
                               alpha=0.15, color='gray', 
                               label=f'WRITE_{i}')
    ax_flat[7].set_ylabel(f'Bit {i}', fontsize=12)
    ax_flat[7].legend(loc='upper right', fontsize=9)
    ax_flat[7].grid(True, alpha=0.3)
    ax_flat[7].set_ylim(-5, 400)
    ax_flat[7].set_xlabel('Time', fontsize=11)
    plt.tight_layout()
    plt.show()