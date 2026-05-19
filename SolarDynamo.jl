module SolarDynamo

export sn, summary_statistics, hann_window

using DelayDiffEq: SDDEProblem, solve
using StochasticDiffEq: EM
using SpecialFunctions: erf
using StaticArrays
import FFTW


# --- Nonlinear function

ftilde(x, Bmin, Bmax) = x/4 * (1 + erf(x^2-Bmin^2)) * (1 - erf(x^2-Bmax^2))


# ---------------------------------
# B-field WITHOUT Jupiter
# ---------------------------------
# See here for the SDDE interface: https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dde_example/

# --- Model : B field

function f(u,h,p,t)     # Drift function
    #  u = [B, dB/dt]
    # du = [dB/dt, d^2B/dt^2]
    τ, T, Nd, sigma, Bmax = p
    hist = h(p, t - T, idxs = 1)    # B[1](t-T)
    inv_τ_2 = inv(τ^2)
    du1 = u[2]
    du2 = -u[1]*inv_τ_2 - 2*u[2]/τ - Nd*inv_τ_2 * ftilde(hist, 1, Bmax)
    SA[du1, du2]
end

function g(u,h,p,t)     # Diffusion function
    τ, T, Nd, sigma, Bmax = p
    du1 = 0
    du2 = Bmax*sigma / (τ * sqrt(τ))
    SA[du1, du2]
end


function bfield(θ, Tsim; kwargs...)

    τ, T, Nd, sigma, Bmax = θ

    # --- define initial values
    u0 = SA[Bmax, 0.0]
    # define inital values [B(t), dB(t)/dt] if t < t0
    # h(p, t) = (Bmax, 0.0)
    # we can speed things up by providing a call by index, see example linked above:
    h(p, t; idxs = nothing) = idxs == 1 ? Bmax : (Bmax, 0.0)

    # use constant lags
    lags = (T, )
    tspan = (0.0, Tsim)

    prob = SDDEProblem(f, g, u0, h, tspan, θ; constant_lags = lags)
    solve(prob, EM(); dt=0.1, saveat=1.0, kwargs...)
end


"""
```
sn(θ; Tobs = 929, Twarmup = 200, kwargs...)
```

Stochastic simulations of the number of sunspots

### Arguments
- `θ`: parameter vector `[τ, T, Nd, sigma, Bmax]`
- `Tobs`: length of the output
- `Twarmup`: length of the warm up period
- `kwargs...`: keyword arguments pased to `solve`. Mostly used ot pass a seed for the random number generator (`seed = 123`).
"""
function sn(θ; Twarmup = 200, Tobs = 929, kwargs...)

    Tsim = Twarmup + Tobs  # Total simulation steps

    sol = bfield(θ, Tsim; kwargs...)

    # square result and get rid of warm up points
    y = map(abs2, sol[1, (Twarmup + 2):end])

    return y
end


function sn_with_noise(θ; Twarmup = 200, Tobs = 200, kwargs...)

    Tsim = Twarmup + Tobs

    sol = bfield(θ, Tsim; save_noise=true, kwargs...)
    
    y = zeros(Tobs)
    noise = zeros(Tobs)
    
    for i in 1:Tobs
        t_start = Twarmup + i - 1.0
        t_end   = Twarmup + i * 1.0
        
        # value of temporal series (B^2) on time t_end
        y[i] = abs2(sol(t_end)[1])
        
        # exact increment of Wiener process on interval [t_start,t_end]
        # sol.W(t) gives the state of noise at time t
        W_start = sol.W(t_start)[2]
        W_end   = sol.W(t_end)[2]
        
        noise[i] = W_end - W_start
    end
    
    return y, noise
end













# --- Model : B field

function f(u,h,p,t)
    # u = [B, dB/dt, W_bare] (Terza variabile per catturare il rumore nudo)
    τ, T, Nd, sigma, Bmax = p
    hist = h(p, t - T, idxs = 1)
   
    inv_τ_2 = inv(τ^2)
    du1 = u[2]
    du2 = -u[1]*inv_τ_2 - 2*u[2]/τ - Nd*inv_τ_2 * ftilde(hist, 1, Bmax)
    du3 = 0.0 # Il rumore nudo non ha dinamica deterministica (drift = 0)
    
    SA[du1, du2, du3]
end

function g(u,h,p,t)
    τ, T, Nd, sigma, Bmax = p
    C = Bmax*sigma / (τ * sqrt(τ))
    
    # Utilizziamo rumore NON diagonale. 
    # Restituiamo una matrice 3x1 che indica come 1 singolo processo Browniano
    # influisce sulle 3 variabili di stato.
    # du1 -> non subisce rumore (0.0)
    # du2 -> subisce il rumore moltiplicato per la costante C
    # du3 -> subisce il rumore puro (1.0). Questa variabile traccerà esattamente il rumore generato!
    SMatrix{3, 1, Float64}(0.0, C, 1.0)
end

function bfield(θ, Tsim; kwargs...)
    τ, T, Nd, sigma, Bmax = θ

    # Inizializziamo lo stato con 3 elementi
    u0 = SA[Bmax, 0.0, 0.0]
    
    # Aggiorniamo la history function per supportare il nuovo vettore di stato
    h(p, t; idxs = nothing) = isnothing(idxs) ? SA[Bmax, 0.0, 0.0] : (idxs == 1 ? Bmax : 0.0)

    lags = (T, )
    tspan = (0.0, Tsim)

    # Indichiamo a SciML la struttura matriciale (non diagonale) del rumore
    nrp = SMatrix{3, 1, Float64}(0.0, 0.0, 0.0)

    prob = SDDEProblem(f, g, u0, h, tspan, θ; constant_lags = lags, noise_rate_prototype = nrp)
    solve(prob, EM(); dt=0.1, saveat=1.0, kwargs...)
end

function sn_with_noise(θ; Twarmup = 200, Tobs = 200, kwargs...)
    Tsim = Twarmup + Tobs  
    
    sol = bfield(θ, Tsim; kwargs...)
    
    y = zeros(Tobs)
    noise = zeros(Tobs)
    
    for i in 1:Tobs
        t_start = Twarmup + i - 1.0
        t_end   = Twarmup + i * 1.0
        
        # y è il quadrato della prima variabile
        y[i] = abs2(sol(t_end)[1])
        
        # Ora il rumore cumulativo è garantito e accessibile all'indice 3
        W_start = sol(t_start)[3]
        W_end   = sol(t_end)[3]
        
        noise[i] = W_end - W_start
    end
    
    return y, noise
end



# -------------
# Summary statistics
# -------------


"""
`hann_window(Tmax)`

Generate a Hann window of size `Tmax`.
"""
hann_window(Tmax) = [0.5*(1 - cos(2.0*π*(t-1)/(Tmax-1))) for t in 1:Tmax]


"""
`summary_statistics(data, window; fourier_range=1:6:120)`

Compute summary statistics for the input data based on the Fourier transform using a given window.

## Arguments

- `data`: The input time series data.
- `window`: Vector of windowing weights to be applied to the data. Defaults to `hann_window(length(data))`.
- `fourier_range`: Indices of the Fourier-transformed components to include in as summary statistics. Defaults to `1:6:120`.
"""
function summary_statistics(data, window=hann_window(length(data));
                            fourier_range=1:6:120)
    fs = FFTW.ifft(window .* data)
    ss = abs.(fs[fourier_range])
    return ss
end


end
